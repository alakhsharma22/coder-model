#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
const int N = 200000;
const int M = 1000;
const int INF = 10000000;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int t;
    cin >> t;

    while (t--) {
        int n, k;
        cin >> n >> k;

        vector<pair<int, int>> a(k);

        for (int i = 0; i < k; i++) {
            int bi, ci;
            cin >> bi >> ci;
            a[i] = {bi, ci};
        }

        sort(a.begin(), a.end());

        int res = 0;
        int cur = 0;

        for (int i = 0; i < k; i++) {
            if (cur < n) {
                res += a[i].second;
                cur++;
            } else {
                break;
            }
        }

        cout << res << endl;
    }

    return 0;
}