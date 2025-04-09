import torch
from torch import nn
import numpy as np
import os
import logging
import subprocess
import tempfile
import resource
import sys
import json
import argparse
from unsloth import FastLanguageModel


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

NUM_CANDIDATES = 8
MAX_LENGTH = 2048
GENERATION_LENGTH = 512
MAX_SOLUTION_LENGTH = 1024
TEMPERATURE = 0.8
TOP_P = 0.9
TOP_K = 50

class RewardFunction:
    """Copied from the training script to ensure compatibility"""
    def __init__(self, lambda_error=0.5, lambda_tle=0.3, lambda_mle=0.2,
                time_limit=2, mem_limit_mb=100):
        self.lambda_error = lambda_error
        self.lambda_tle = lambda_tle
        self.lambda_mle = lambda_mle
        self.time_limit = time_limit
        self.mem_limit_mb = mem_limit_mb
    
    def limit_memory(self, mem_limit_bytes):
        resource.setrlimit(resource.RLIMIT_AS, (mem_limit_bytes, mem_limit_bytes))
    
    def compile_cpp_code(self, cpp_code, filename_prefix="temp_cpp"):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False, prefix=filename_prefix) as src_file:
            src_file.write(cpp_code)
            cpp_filename = src_file.name
        
        exe_filename = cpp_filename.replace(".cpp", ".out")
        compile_command = ["g++", cpp_filename, "-o", exe_filename]
        
        compile_result = subprocess.run(compile_command, capture_output=True, text=True)
        
        if compile_result.returncode != 0:
            compile_error = compile_result.stderr
            os.remove(cpp_filename)
            return None, compile_error
        else:
            return exe_filename, None
    
    def run_cpp_executable(self, exe_filename, test_input):
        error_flag = False
        tle_flag = False
        mle_flag = False
        
        mem_limit_bytes = self.mem_limit_mb * 1024 * 1024
        
        try:
            result = subprocess.run(
                [exe_filename],
                input=test_input.encode('utf-8'),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.time_limit,
                preexec_fn=lambda: self.limit_memory(mem_limit_bytes)
            )
            stdout = result.stdout.decode('utf-8').strip()
            stderr = result.stderr.decode('utf-8').strip()
            
            if result.returncode != 0:
                error_flag = True
            
            if "std::bad_alloc" in stderr or "MemoryError" in stderr:
                mle_flag = True
                
        except subprocess.TimeoutExpired:
            stdout = ""
            stderr = "TimeoutExpired"
            tle_flag = True
            error_flag = True
        
        return stdout, stderr, error_flag, tle_flag, mle_flag
    
    def evaluate(self, cpp_code, test_cases):
        exe_filename, compile_error = self.compile_cpp_code(cpp_code)
        if exe_filename is None:
            return -self.lambda_error, {"compile_error": compile_error}
        
        passed_count = 0
        error_occurred = False
        tle_occurred = False
        mle_occurred = False
        test_details = []
        
        for idx, test in enumerate(test_cases):
            stdout, stderr, error_flag, tle_flag, mle_flag = self.run_cpp_executable(
                exe_filename, test["input"])
            
            passed = (stdout == test["expected"])
            if passed:
                passed_count += 1
            if error_flag:
                error_occurred = True
            if tle_flag:
                tle_occurred = True
            if mle_flag:
                mle_occurred = True
            
            test_details.append({
                "test_index": idx,
                "passed": passed,
                "stdout": stdout,
                "stderr": stderr,
                "error_flag": error_flag,
                "tle_flag": tle_flag,
                "mle_flag": mle_flag,
            })
        
        fraction_passed = passed_count / len(test_cases) if test_cases else 0.0
        
        reward = (fraction_passed
                 - self.lambda_error * (1 if error_occurred else 0)
                 - self.lambda_tle * (1 if tle_occurred else 0)
                 - self.lambda_mle * (1 if mle_occurred else 0))
        
        try:
            os.remove(exe_filename)
            src_filename = exe_filename.replace(".out", ".cpp")
            if os.path.exists(src_filename):
                os.remove(src_filename)
        except Exception as e:
            print("Cleanup error:", e, file=sys.stderr)
        
        details = {
            "passed_count": passed_count,
            "total_tests": len(test_cases),
            "error_occurred": error_occurred,
            "tle_occurred": tle_occurred,
            "mle_occurred": mle_occurred,
            "test_details": test_details
        }
        
        return reward, details

class CodeInference:
    def __init__(self, 
                 model_path, 
                 device="cuda", 
                 num_candidates=NUM_CANDIDATES,
                 temperature=TEMPERATURE,
                 top_p=TOP_P, 
                 top_k=TOP_K,
                 max_length=MAX_LENGTH,
                 generation_length=GENERATION_LENGTH,
                 load_in_4bit=True,
                 is_lora_model=False):
        self.model, self.tokenizer = self.load_model(
            model_path, 
            device, 
            load_in_4bit=load_in_4bit,
            is_lora_model=is_lora_model
        )
        self.reward_function = RewardFunction()
        self.device = device
        self.num_candidates = num_candidates
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_length = max_length
        self.generation_length = generation_length
        self.max_solution_length = MAX_SOLUTION_LENGTH
        
        self._fix_lora_dtypes()
    
    def _fix_lora_dtypes(self):
        """Ensure all LoRA parameters have the same dtype as the base model."""
        try:
            model_dtype = next(self.model.parameters()).dtype
            for name, param in self.model.named_parameters():
                if 'lora' in name.lower() and param.dtype != model_dtype:
                    logger.info(f"Converting {name} from {param.dtype} to {model_dtype}")
                    param.data = param.data.to(model_dtype)
        except Exception as e:
            logger.error(f"Error fixing LoRA dtypes: {e}")
    
    def load_model(self, model_path, device, load_in_4bit=True, is_lora_model=False):
        """Load either a merged model or a LoRA model depending on is_lora_model flag"""
        logger.info(f"Loading model from {model_path} (LoRA: {is_lora_model})")
        
        common_args = {
            "model_name": model_path,
            "max_seq_length": MAX_LENGTH,
            "load_in_4bit": load_in_4bit,
        }
        
        if is_lora_model:
            model, tokenizer = FastLanguageModel.from_pretrained(**common_args)            
            model = FastLanguageModel.get_peft_model(
                model,
                r=32,  
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
                lora_alpha=32,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing=False,
                random_state=3407,
                use_rslora=False,
            )
        else:
            model, tokenizer = FastLanguageModel.from_pretrained(**common_args)
        
        inference_model = FastLanguageModel.for_inference(model)
        inference_model.eval()
        
        return inference_model, tokenizer
    
    def format_problem(self, problem):
        prompt = f"""
You are an expert competitive programmer. Your task is to solve the following programming problem.
Write ONLY the complete C++ solution code without explanations, notes, or comments about your approach.
Do not include markdown formatting tags in your response.

Problem Name: {problem['Name']}

Problem Statement:
{problem['Statement']}

Input Format:
{problem['Input Format']}

Output Format:
{problem['Output Format']}

Constraints:
{problem['Constraints']}

Example Input:
{problem['Example Input']}

Example Output:
{problem['Example Output']}

Now provide only a correct and efficient C++ solution. Your solution must:
1. Include all necessary headers
2. Have a main() function that reads inputs and outputs the result
3. Be complete and self-contained
4. Compile without errors
5. Have no explanations - just code

Begin your solution with these starter lines:
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {{
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
}}
    
"""
        return prompt
    
    def validate_cpp_code(self, code):
        """Validates if the string looks like compilable C++ code"""
        if len(code) > 200:
            logger.debug(f"Validating code (showing first 200 chars): {code[:200]}...")
        else:
            logger.debug(f"Validating code: {code}")
        if "```" in code:
            code = code.replace("```", "")
            logger.debug("Removed markdown artifacts from code")
        has_include = '#include' in code
        has_main = 'int main' in code
        has_opening_brace = '{' in code
        has_closing_brace = '}' in code
        
        modified_code = code
        
        if not has_include and not code.strip().startswith('#include'):
            modified_code = "#include <iostream>\n#include <vector>\n#include <algorithm>\n" + modified_code
            logger.debug("Added missing includes")
            has_include = True
            
        if not "using namespace std" in modified_code:
            lines = modified_code.split('\n')
            include_indices = [i for i, line in enumerate(lines) if line.strip().startswith('#include')]
            if include_indices:
                last_include = max(include_indices)
                lines.insert(last_include + 1, "using namespace std;")
                modified_code = '\n'.join(lines)
                logger.debug("Added 'using namespace std;'")
        
        if not (has_include and has_main and has_opening_brace and has_closing_brace):
            logger.warning("Generated code missing critical C++ elements that couldn't be automatically fixed")
            code_content = ""
            if "main" in code:
                main_start = code.find("main")
                open_brace = code.find("{", main_start)
                if open_brace > -1:
                    depth = 1
                    pos = open_brace + 1
                    while pos < len(code) and depth > 0:
                        if code[pos] == '{':
                            depth += 1
                        elif code[pos] == '}':
                            depth -= 1
                        pos += 1
                    
                    if depth == 0:
                        main_content = code[open_brace+1:pos-1].strip()
                        code_content = f"    // Salvaged logic from generated code:\n    {main_content}\n"
            
            return False, f"""
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {{
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
{code_content}
    // Invalid code generated - using fallback
    return 0;
}}
"""
        lines = modified_code.split('\n')
        clean_lines = []
        skip_section = False
        
        for line in lines:
            lower_line = line.lower().strip()

            if lower_line.startswith(("note:", "explanation:", "//note", "// note", "/*", "// explanation", "// approach")):
                skip_section = True
                continue

            if skip_section and (lower_line == "*/" or not lower_line):
                skip_section = False
                continue
                
            if not skip_section:
                clean_lines.append(line)
        
        cleaned_code = '\n'.join(clean_lines)

        if "return 0" not in cleaned_code and "return 0;" not in cleaned_code:
            last_brace = cleaned_code.rfind('}')
            if last_brace > 0:
                cleaned_code = cleaned_code[:last_brace] + "\n    return 0;\n}" + cleaned_code[last_brace+1:]
                logger.debug("Added missing 'return 0;' statement")
        
        return True, cleaned_code
    
    def parse_solution(self, generated_text):
        try:
            if len(generated_text) > 200:
                logger.debug(f"Generated text (first 200 chars): {generated_text[:200]}...")
            else:
                logger.debug(f"Generated text: {generated_text}")
                
            solution = ""
            
            if "```cpp" in generated_text and "```" in generated_text[generated_text.find("```cpp") + 6:]:
                start_idx = generated_text.find("```cpp") + 6
                end_idx = generated_text.find("```", start_idx)
                solution = generated_text[start_idx:end_idx].strip()
                logger.debug("Extracted code from markdown block")
            elif "```c++" in generated_text and "```" in generated_text[generated_text.find("```c++") + 6:]:
                start_idx = generated_text.find("```c++") + 6
                end_idx = generated_text.find("```", start_idx)
                solution = generated_text[start_idx:end_idx].strip()
                logger.debug("Extracted code from c++ markdown block")
            elif "```" in generated_text and "```" in generated_text[generated_text.find("```") + 3:]:

                start_idx = generated_text.find("```") + 3
                end_idx = generated_text.find("```", start_idx)
                code_block = generated_text[start_idx:end_idx].strip()
                if "#include" in code_block and "main" in code_block:
                    solution = code_block
                    logger.debug("Extracted code from generic markdown block")
            
            if not solution:
                starter_code = "#include <iostream>\n#include <vector>\n#include <algorithm>\nusing namespace std;\n\nint main() {\n    ios_base::sync_with_stdio(false);\n    cin.tie(NULL);"
                if starter_code in generated_text:
                    start_idx = generated_text.find(starter_code)
                    explanatory_markers = ["Note:", "Explanation:", "// Note", "/* Note", "// Explanation"]
                    end_idx = len(generated_text)
                    for marker in explanatory_markers:
                        marker_pos = generated_text.find(marker, start_idx)
                        if marker_pos > 0 and marker_pos < end_idx:
                            end_idx = marker_pos
                    
                    solution = generated_text[start_idx:end_idx].strip()
                    logger.debug("Extracted code using starter code detection")
                else:
                    lines = generated_text.split('\n')
                    code_lines = []
                    in_code = False
                    ignore_line = False
                    
                    for line in lines:
                        if "```" in line:
                            continue
                            
                        if line.strip().lower().startswith(("note:", "explanation:", "comment:", "// note", "// explanation")):
                            ignore_line = True
                            continue

                        if ignore_line and not line.strip():
                            ignore_line = False
                            continue
                        
                        if ignore_line:
                            continue
                        if not in_code and (line.strip().startswith('#include') or 
                                        line.strip().startswith('using namespace') or
                                        "int main" in line):
                            in_code = True
                        
                        if in_code:
                            code_lines.append(line)
                    
                    solution = '\n'.join(code_lines)
                    logger.debug("Extracted code using line-by-line parsing")

            if not any(marker in solution for marker in ['#include', 'int main']):
                code_fragments = []
                in_algorithm = False
                
                for line in generated_text.split('\n'):
                    stripped = line.strip()
                    if not in_algorithm and (
                        stripped.startswith('for(') or 
                        stripped.startswith('for ') or
                        stripped.startswith('while(') or
                        stripped.startswith('while ') or
                        stripped.startswith('if(') or
                        stripped.startswith('if ') or
                        "vector<" in stripped or
                        "sort(" in stripped or
                        "return " in stripped
                    ):
                        in_algorithm = True
                    
                    if in_algorithm:
                        code_fragments.append(line)
                        if stripped.startswith('//') and any(word in stripped.lower() for word in ['note', 'explain', 'approach']):
                            in_algorithm = False
                
                if code_fragments:
                    algorithm_code = '\n'.join(code_fragments)
                    solution = f"""
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {{
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    // Salvaged algorithm code:
{algorithm_code}
    
    return 0;
}}
"""
                    logger.debug("Created solution using salvaged algorithm fragments")
                else:
                    solution = """
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    // Fallback empty solution
    return 0;
}
"""
                    logger.debug("Using fallback solution - no valid code extracted")
            
            if len(solution) > self.max_solution_length:
                solution = solution[:self.max_solution_length]
                logger.debug(f"Truncated solution to {self.max_solution_length} characters")
                
            return solution
                
        except Exception as e:
            logger.warning(f"Error parsing solution: {e}")
            return """
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    // Error in parsing
    return 0;
}
"""
    
    def create_test_cases(self, problem):
        test_cases = []
        
        test_cases.append({
            "input": problem["Example Input"],
            "expected": problem["Example Output"]
        })
        
        input_examples = problem["Example Input"].split("Example Input")
        output_examples = problem["Example Output"].split("Example Output")
        
        if len(input_examples) > 1 and len(output_examples) > 1:
            for i in range(1, min(len(input_examples), len(output_examples))):
                test_cases.append({
                    "input": input_examples[i].strip(),
                    "expected": output_examples[i].strip()
                })
        
        return test_cases
    
    def generate_solutions(self, problem, verbose=True):
        prompt = self.format_problem(problem)
        candidates = []
        
        if verbose:
            logger.info(f"Generating {self.num_candidates} candidate solutions for problem: {problem['Name']}")
        
        for i in range(self.num_candidates):
            try:
                with torch.no_grad():
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                    output_ids = self.model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=min(512, self.generation_length),
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
                    generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    
                    cpp_code = self.parse_solution(generated_text)
                    is_valid, cleaned_code = self.validate_cpp_code(cpp_code)
                    
                    candidates.append(cleaned_code)
                    
                    if verbose:
                        logger.info(f"Generated candidate {i+1}/{self.num_candidates}")
                    
            except Exception as e:
                logger.error(f"Generation error: {e}")
                candidates.append("""
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    // Fallback solution
    return 0;
}
                """.strip()[:self.max_solution_length])
        
        return candidates
    
    def evaluate_candidates(self, candidates, problem, verbose=True):
        test_cases = self.create_test_cases(problem)
        rewards = []
        details = []
        
        if verbose:
            logger.info(f"Evaluating {len(candidates)} candidates")
        
        for i, candidate in enumerate(candidates):
            is_valid, cleaned_code = self.validate_cpp_code(candidate)
            if not is_valid and verbose:
                logger.warning("Invalid C++ code detected, using fallback")
                
            reward, detail = self.reward_function.evaluate(cleaned_code, test_cases)
            rewards.append(reward)
            details.append(detail)

            if verbose:
                if 'compile_error' in detail:
                    logger.info(f"Candidate {i+1}: Reward = {reward:.4f}, Compilation Error")
                else:
                    logger.info(f"Candidate {i+1}: Reward = {reward:.4f}, Passed {detail['passed_count']}/{detail['total_tests']} tests")
        
        return rewards, details
    
    def get_best_solution(self, problem, verbose=True):
        candidates = self.generate_solutions(problem, verbose)
        rewards, details = self.evaluate_candidates(candidates, problem, verbose)
        
        best_idx = np.argmax(rewards)
        best_candidate = candidates[best_idx]
        best_reward = rewards[best_idx]
        best_details = details[best_idx]
        
        if verbose:
            logger.info(f"Best solution has reward {best_reward:.4f}")
        
        return {
            "solution": best_candidate,
            "reward": best_reward,
            "details": best_details,
            "all_candidates": candidates,
            "all_rewards": rewards,
            "all_details": details
        }

def load_problem_from_json(json_path):
    """Load a problem description from a JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Generate C++ solutions for competitive programming problems")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the trained model (either merged model or LoRA adapter)")
    parser.add_argument("--problem_file", type=str, required=True, 
                        help="JSON file containing the problem description")
    parser.add_argument("--output_file", type=str, default="solution.cpp", 
                        help="File to save the generated solution")
    parser.add_argument("--candidates", type=int, default=NUM_CANDIDATES, 
                        help="Number of candidate solutions to generate")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, 
                        help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=TOP_P, 
                        help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=TOP_K, 
                        help="Top-k sampling parameter")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to run inference on (cuda or cpu)")
    parser.add_argument("--is_lora", action="store_true", 
                        help="Specify if the model is a LoRA adapter (not a merged model)")
    parser.add_argument("--no_4bit", action="store_true", 
                        help="Disable 4-bit quantization for model loading")
    parser.add_argument("--verbose", action="store_true", 
                        help="Enable verbose logging for debugging")
    parser.add_argument("--save_all", action="store_true", 
                        help="Save all candidate solutions, not just the best one")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    problem = load_problem_from_json(args.problem_file)

    inference = CodeInference(
        model_path=args.model_path,
        device=args.device,
        num_candidates=args.candidates,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        load_in_4bit=not args.no_4bit,
        is_lora_model=args.is_lora
    )
    
    result = inference.get_best_solution(problem, verbose=True)
    
    with open(args.output_file, 'w') as f:
        f.write(result["solution"])
    
    logger.info(f"Best solution saved to {args.output_file}")
    
    if args.save_all:
        logger.info("Saving all candidate solutions")
        for i, candidate in enumerate(result["all_candidates"]):
            candidate_file = f"{args.output_file}.candidate_{i+1}.cpp"
            reward = result["all_rewards"][i]
            with open(candidate_file, 'w') as f:
                f.write(f"// Candidate {i+1}, Reward: {reward:.4f}\n\n")
                f.write(candidate)
            logger.info(f"Candidate {i+1} saved to {candidate_file}")

    results_file = f"{args.output_file}.results.json"
    
    result_json = {
        "problem_name": problem["Name"],
        "best_reward": result["reward"]
    }
    
    if 'compile_error' in result["details"]:
        result_json.update({
            "compile_error": result["details"]["compile_error"],
            "passed_tests": 0,
            "total_tests": 0,
            "error_occurred": True,
            "tle_occurred": False,
            "mle_occurred": False
        })
    else:
        result_json.update({
            "passed_tests": result["details"]["passed_count"],
            "total_tests": result["details"]["total_tests"],
            "error_occurred": result["details"]["error_occurred"],
            "tle_occurred": result["details"]["tle_occurred"],
            "mle_occurred": result["details"]["mle_occurred"],
            "test_details": result["details"]["test_details"]
        })
    
    with open(results_file, 'w') as f:
        json.dump(result_json, f, indent=2)
    
    logger.info(f"Detailed results saved to {results_file}")

if __name__ == "__main__":
    main()