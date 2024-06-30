# different posssibilty -
# codeforces_608_A
# difficult for chatgpt for custom input parsing 

#1
arr = list(map(int, input().split()))
---
arr = [int(input()) for _ in range(n)]
# or in seperate line 


#2
# atcoder_ABC043_B.py
# errors-
# also class definition ignored and considered competitive type input 
'''
1.StringTokenizer
def string(self):
        if self.tokenizer is None or not self.tokenizer.hasMoreTokens():
            self.tokenizer = StringTokenizer(self.reader.readline())
        return self.tokenizer.nextToken()
->
    def string(self):
        if self.tokens is None or len(self.token)==0:
            self.tokens = self.reader.readline().split()
        return self.tokens.pop(0)

2.out.println(''.join(d))
-> 
print(''.join(d), file=out)

3. keywords -"in"

'''
import sys
class BUnhappyHackingABCEdit:
    def solve(self, testNumber, in_, out):
        s = in_.string()
        d = []
        for c in s:
            if c == '0':
                d.append('0')
            elif c == '1':
                d.append('1')
            elif c == 'B':
                if d:
                    d.pop()
        print(''.join(d), file=out)


class LightScanner:
    def __init__(self, in_):
        self.reader = sys.stdin
        self.tokens = None

    def string(self):
        if self.tokens is None or len(self.token)==0:
            self.tokens = self.reader.readline().split()
        return self.tokens.pop(0)

def main():
    in_ = LightScanner(sys.stdin)
    out = sys.stdout
    solver = BUnhappyHackingABCEdit()
    solver.solve(1, in_, out)

if __name__ == '__main__':
    main()

##3.
# codeforces_37_A 
import sys

class FastScanner:
    def __init__(self):
        self.buf = sys.stdin.readline
        self.tokens = None

    def has_next(self):
        while self.tokens is None or len(self.tokens) == 0:
            self.tokens = self.buf().split()
        #some warning need if no input or below line 
        if (self.tokens is None or len(self.tokens) == 0):
            return False
        return True

    def next_int(self):
        if not self.has_next():
            return None
        val = self.tokens.pop(0)
        return int(val)

    def next_str(self):
        if not self.has_next():
            return None
        val = self.tokens.pop(0)
        return val

def main():
    input = FastScanner()
    n = input.next_int()
    map = {}
    for i in range(n):
        val = input.next_int()
        map[val] = map.get(val, 0) + 1
    max = -1
    for entry in map.items():
        value = entry[1]
        max = max if max > value else value
    print(max, len(map))

if __name__ == "__main__":
    main()


