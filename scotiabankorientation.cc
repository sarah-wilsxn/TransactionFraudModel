/*
At a high school with one thousand lockers and one thousand students, on the first day of class the principal asks the first student to go
to every locker and open it. Then he has the second student go to every second locker and close it. The third goes to every third locker
and, if it is closed, he opens it, and if it is open, he closes it. The fourth student does this to every fourth locker, and so on. After the
process is completed with the thousandth student, how many lockers are open?
*/

#include <iostream>
#include <vector>
using namespace std;

int soln(int n) {
  // 0 = closed, 1 = open
    vector<int> v(n, 0);
    for (int i = 1; i <= n; i++){
       for (int j = i; j <= n; j += i){
           v[j] = (v[j] == 0) ? 1 : 0; 
       }
    }
    int open = 0;
    for (auto x : v){
        if (x == 1) open++;
    }
    return open;
}

int main() {
  cout << soln(1000) << endl;
}
