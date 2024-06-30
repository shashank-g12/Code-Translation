import json
import os
from abc import ABC, abstractmethod
from typing import List
from warnings import warn
import glob


'''
# approaches in compose_prompt
# -EOS debugging and clean
    <END-OF-CODE>
# could be multiple eos in outputs, better pick minimum one
# class HFTorchDecoder(DecoderBase):
    -starcoder (StarCoderInfill  or VLlmDecoder())

-granite models-
    -llama and starcoder similar 
    -EOS - <|endoftext|>
    -VLlmDecoder like codellama ?

self.context_window_length =max_length #4090
'''

# cache_dir = os.getcwd() + "/huggingface"
cache_dir = "/home/cse/dual/cs5190439/MTP1/codetlingua"+ "/huggingface"
os.environ["TRANSFORMERS_CACHE"] = cache_dir

import anthropic
import anthropic_request
import google.generativeai as genai

import openai
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from openai_request import make_auto_request
from vllm import LLM, SamplingParams

EOS = ["<|endoftext|>", "<|endofmask|>", "</s>","<END-OF-CODE>",("\n"*11)]

#J2P
EOS.append("4. Java")
EOS.append("###$###")

# P2J
# EOS+=["###$###"]
EOS+=["4. Python"]

# autocot1
EOS+=["4. Whole Java Code:"]

# function to build prompt accroding to various model instruction formats
def compose_prompt(prompt_type: str, source_lang: str, target_lang: str, code: str) -> str:

    # base 
    prompt = f"{source_lang}:\n{code}\n\nTranslate the above {source_lang} code to {target_lang} and end with comment \"<END-OF-CODE>\".\n\n{target_lang}:\n"
    #use source_lang cond with autocot2d
    #final_v1_1 similar to v1

    if prompt_type=="program" and source_lang=="Python":## v6 -program with v1_1 and for variables removed 
        prompt='''###$###
1. Python Code:
l = [ ]
limit = 10000000000
def gen ( number , four , seven ) :
    global debug
    debug = True
    status = "bad"
    if ( number > limit ) :
        return
    if ( number > 0 and four == seven ) :
        l.append ( number )
    gen ( number * 10 + 4 , four + 1 , seven )
    gen ( number * 10 + 7 , four , seven + 1 )

debug= False
status = "good"
def main ( ) :
    gen ( 0 , 0 , 0 )
    l.sort ( )
    n = int ( input ( ) )
    ans = 0
    for val in l :
        if ( val >= n ) :
            ans = val
            break
    print ( ans )

1. Sample Input:
9

1. Expected Output:
47


1. Steps: Let's think step by step.

Step 1: First of all identify global variables and their types. Global variables are variables that are not inside scope of function and class declarations, present in the above Python Code-
Global variables: 
l, limit, debug, status,

Step 2: Identify classes and functions declarations present in the above Python Code-
Identified classes and functions declarations in the Python code:
def gen ( number , four , seven ) :

def main ( ) :

Step 3: Considering Step 1 and Step 2, the corresponding Java Code should have declarations of these classes, functions and Global variables -
Identified classes, functions and Global variables declarations in the Java code:
public class Main {
    static ArrayList<Long> l;
    static long limit;
    static boolean debug;
    static String status;

    public static void gen(long number, int four, int seven);

    public static void main ( String [ ] args );

}

Step 4: Complete translation of the above Python code to Java code by considering identified classes, functions and Global variables declarations in Step 3. Consider above sample input and output format information to handle input and output related code properly in the generated Java code. While generating the Java code, keep track of required library imports to be added in the Java Code. Make sure your generated code is syntactically correct-
1. Java Code:
import java.util.*;

public class Main{
    static ArrayList<Long> l;
    static long limit;
    static boolean debug;
    static String status;

    public static void gen(long number, int four, int seven) {
        Main.debug = True;
        String status = "bad";
        if (number > limit) return;
        if (number > 0 && four == seven) {
            l.add(number);
        }
        gen(number * 10 + 4, four + 1, seven);
        gen(number * 10 + 7, four, seven + 1);
    }

    public static void main(String[] args) {
        l = new ArrayList<>();
        limit = 10000000000L;
        debug= False;
        status = "good";

        gen(0, 0, 0);
        Collections.sort(l);
        Scanner sc = new Scanner(System.in);
        long n = sc.nextLong();
        long ans = 0;
        for (long val : l) {
            if (val >= n) {
                ans = val;
                break;
            }
        }
        System.out.println(ans);
    }
}

###$###
2. Python Code:
class BIT :
    def __init__ ( self , N ) :
        self.size = N
        self.tree = [ 0 ] * ( N + 1 )
        self.depth = n.bit_length ( )
    def _bitsum ( self , i ) :
        ret = 0
        while i :
            ret += self.tree [ i ]
            i ^= i & - i
        return ret
    def bitsum ( self , l , r = None ) :
        if r is None :
            return self._bitsum ( l )
        else :
            return self._bitsum ( r ) - self._bitsum ( l )
    def bitadd ( self , i , x ) :
        i += 1
        while i <= self.size :
            self.tree [ i ] += x
            i += i & - i
        return
n = int ( input ( ) )
m = n * ( n + 1 ) // 4
a = list ( map ( int , input ( ).split ( ) ) )
d = dict ( )
_a = sorted ( set ( a + [ 0 ] ) )
for i , x in enumerate ( _a ) :
    d [ x ] = i
a = [ d [ x ] for x in a ]
def check ( X ) :
    b = [ 0 ] + [ ( y >= X ) * 2 - 1 for y in a ]
    for i in range ( n ) :
        b [ i + 1 ] += b [ i ]
    c = min ( b )
    b = [ x - c for x in b ]
    bit = BIT ( max ( b ) + 2 )
    ans = 0
    for x in b :
        ans += bit.bitsum ( x + 1 )
        bit.bitadd ( x , 1 )
    return ans >= m
t = [ len ( _a ) , 0 ]
while t [ 0 ] - t [ 1 ] > 1 :
    mid = ( t [ 0 ] + t [ 1 ] ) // 2
    t [ check ( mid ) ] = mid
print ( _a [ t [ 1 ] ] )

2. Sample Input:
1
1

2. Expected Output:
1


2. Steps: Let's think step by step.

Step 1: First of all identify global variables and their types. Global variables are variables that are not inside scope of function and class declarations, present in the above Python Code-
Global variables:
t, d, n, _a, mid, m, a,

Step 2: Identify classes and functions declarations present in the above Python Code-
Identified classes and functions declarations in the Python code:
class BIT :
    def __init__ ( self , N ) :
        self.size = N
        self.tree = [ 0 ] * ( N + 1 )
        self.depth = N.bit_length ( )

    def _bitsum ( self , i ) :

    def bitsum ( self , l , r = None ) :

    def bitadd ( self , i , x ) :

def check ( X ) :


Step 3: Considering Step 1 and Step 2, the corresponding Java Code should have declarations of these classes, functions and Global variables -
Identified classes, functions and Global variables declarations in the Java code:
public class Main {
    static int[] t;
    static Map<Integer, Integer> d;
    static int n;
    static List<Integer> _a;
    static int mid;
    static long m;
    static int[] a;

    public static class BIT {
        public int size;
        public long [] tree;
        public int depth;

        public BIT(int N);

        public long _bitsum(int i);

        public long bitsum(int l, int r);

        public long bitsum(int l);

        public void bitadd(int i, int x);

        }

    public static boolean check(int X);

    public static void main ( String [ ] args );

}

Step 4: Complete translation of the above Python code to Java code by considering identified classes, functions and Global variables declarations in Step 3. Consider above sample input and output format information to handle input and output related code properly in the generated Java code. While generating the Java code, keep track of required library imports to be added in the Java Code. Make sure your generated code is syntactically correct-
2. Java Code:
import java.util.*;
import java.io.*;
import java.util.stream.Collectors;

public class Main{
    static int[] t;
    static Map<Integer, Integer> d;
    static int n;
    static List<Integer> _a;
    static int mid;
    static long m;
    static int[] a;

    public static class BIT {
        public int size;
        public long [] tree;
        public int depth;

        public BIT(int N) {
            this.size = N;
            this.tree = new long[N + 1];
            this.depth = Integer.toBinaryString(N).length();
        }

        public long _bitsum(int i) {
            long ret = 0;
            while (i > 0) {
                ret += this.tree[i];
                i ^= i & -i;
            }
            return ret;
        }

        public long bitsum(int l, int r) {
            return this._bitsum(r) - this._bitsum(l);
        }

        public long bitsum(int l) {
            return this._bitsum(l);
        }

        public void bitadd(int i, int x) {
            i += 1;
            while (i <= this.size) {
                this.tree[i] += x;
                i += i & -i;
            }
            return;
        }
    }


    public static boolean check(int X) {
        int[] b = new int[n + 1];
        b[0] = 0;
        for (int i = 0; i < n; i++) {
            b[i + 1] = (a[i] >= X ? 2 : 0) - 1;
        }
        for (int i = 0; i < n; i++) {
            b[i + 1]+= b[i];
        }
        int c = Arrays.stream(b).min().getAsInt();
        for (int i = 0; i < b.length; i++) {
            b[i] -= c;
        }
        BIT bit = new BIT(Arrays.stream(b).max().getAsInt() + 2);
        long ans = 0;
        for (int x : b) {
            ans += (long) bit.bitsum(x + 1);
            bit.bitadd(x, 1);
        }
        return ans >= m;
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        n = scanner.nextInt();
        m = (long) n * (n + 1) / 4;
        a = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = scanner.nextInt();
        }
        d = new HashMap<>();
        Set<Integer> set = new HashSet<>();
        for (int x : a) {
            set.add(x);
        }
        set.add(0);
        _a = new ArrayList<>(set);
        Collections.sort(_a);
        for (int i = 0; i < _a.size(); i++) {
            d.put(_a.get(i), i);
        }
        for (int i = 0; i < a.length; i++) {
            a[i] = d.get(a[i]);
        }

        t = new int[]{ _a.size(), 0 };

        while (t[0] - t[1] > 1) {
            mid = (t[0] + t[1]) / 2;
            if (check(mid)) {
                t[1] = mid;
            } else {
                t[0] = mid;
            }
        }
        System.out.println(_a.get(t[1]));
    }
}

###$###
3. Python Code:
'''
        prompt+=code
    elif prompt_type=="autocot2d" and source_lang=="Python":#autocot2d_p2j
        prompt='''###$###
1. Python Code:
l = [ ]
limit = 10000000000
def gen ( number , four , seven ) :
    global debug
    debug = True
    status = "bad"
    if ( number > limit ) :
        return
    if ( number > 0 and four == seven ) :
        l.append ( number )
    gen ( number * 10 + 4 , four + 1 , seven )
    gen ( number * 10 + 7 , four , seven + 1 )

debug= False
status = "good"
def main ( ) :
    gen ( 0 , 0 , 0 )
    l.sort ( )
    n = int ( input ( ) )
    ans = 0
    for val in l :
        if ( val >= n ) :
            ans = val
            break
    print ( ans )

1. Sample Input:
9

1. Expected Output:
47


1. Steps: Let's think step by step.

Step 1: First of all identify global variables and their types. Global variables are variables that are not inside scope of function and class declarations, present in the above Python Code-
Global variables: l, limit, debug, status

Step 2: Identify classes and functions declarations present in the above Python Code-
Identified classes and functions declarations in the Python code:
def gen ( number , four , seven ) :

def main ( ) :

Step 3: Considering Step 1 and Step 2, the corresponding Java Code should have declarations of these classes, functions and Global variables -
Identified classes, functions and Global variables declarations in the Java code:
public class Main {
    static ArrayList<Long> l;
    static long limit;
    static boolean debug;
    static String status;

    public static void gen(long number, int four, int seven);

    public static void main ( String [ ] args );

}

Step 4: Complete translation of the above Python code to Java code by considering identified classes, functions and Global variables declarations in Step 3. Consider above sample input and output format information to handle input and output related code properly in the generated Java code. While generating the Java code, keep track of required library imports to be added in the Java Code. Make sure your generated code is syntactically correct-
1. Java Code:
import java.util.*;

public class Main{
    static ArrayList<Long> l;
    static long limit;
    static boolean debug;
    static String status;

    public static void gen(long number, int four, int seven) {
        Main.debug = True;
        String status = "bad";
        if (number > limit) return;
        if (number > 0 && four == seven) {
            l.add(number);
        }
        gen(number * 10 + 4, four + 1, seven);
        gen(number * 10 + 7, four, seven + 1);
    }

    public static void main(String[] args) {
        l = new ArrayList<>();
        limit = 10000000000L;
        debug= False;
        status = "good";

        gen(0, 0, 0);
        Collections.sort(l);
        Scanner sc = new Scanner(System.in);
        long n = sc.nextLong();
        long ans = 0;
        for (long val : l) {
            if (val >= n) {
                ans = val;
                break;
            }
        }
        System.out.println(ans);
    }
}

###$###
2. Python Code:
class BIT :
    def __init__ ( self , N ) :
        self.size = N
        self.tree = [ 0 ] * ( N + 1 )
        self.depth = n.bit_length ( )
    def _bitsum ( self , i ) :
        ret = 0
        while i :
            ret += self.tree [ i ]
            i ^= i & - i
        return ret
    def bitsum ( self , l , r = None ) :
        if r is None :
            return self._bitsum ( l )
        else :
            return self._bitsum ( r ) - self._bitsum ( l )
    def bitadd ( self , i , x ) :
        i += 1
        while i <= self.size :
            self.tree [ i ] += x
            i += i & - i
        return
n = int ( input ( ) )
m = n * ( n + 1 ) // 4
a = list ( map ( int , input ( ).split ( ) ) )
d = dict ( )
_a = sorted ( set ( a + [ 0 ] ) )
for i , x in enumerate ( _a ) :
    d [ x ] = i
a = [ d [ x ] for x in a ]
def check ( X ) :
    b = [ 0 ] + [ ( y >= X ) * 2 - 1 for y in a ]
    for i in range ( n ) :
        b [ i + 1 ] += b [ i ]
    c = min ( b )
    b = [ x - c for x in b ]
    bit = BIT ( max ( b ) + 2 )
    ans = 0
    for x in b :
        ans += bit.bitsum ( x + 1 )
        bit.bitadd ( x , 1 )
    return ans >= m
t = [ len ( _a ) , 0 ]
while t [ 0 ] - t [ 1 ] > 1 :
    mid = ( t [ 0 ] + t [ 1 ] ) // 2
    t [ check ( mid ) ] = mid
print ( _a [ t [ 1 ] ] )

2. Sample Input:
1
1

2. Expected Output:
1


2. Steps: Let's think step by step.

Step 1: First of all identify global variables and their types. Global variables are variables that are not inside scope of function and class declarations, present in the above Python Code-
Global variables: n, m, a, d, _a, t

Step 2: Identify classes and functions declarations present in the above Python Code-
Identified classes and functions declarations in the Python code:
class BIT :
    def __init__ ( self , N ) :
        self.size = N
        self.tree = [ 0 ] * ( N + 1 )
        self.depth = N.bit_length ( )

    def _bitsum ( self , i ) :

    def bitsum ( self , l , r = None ) :

    def bitadd ( self , i , x ) :

def check ( X ) :


Step 3: Considering Step 1 and Step 2, the corresponding Java Code should have declarations of these classes, functions and Global variables -
Identified classes, functions and Global variables declarations in the Java code:
public class Main {
    static int n;
    static long m;
    static int[] a;
    static Map<Integer, Integer> d;
    static List<Integer> _a;
    static int[] t;

    public static class BIT {
        public int size;
        public long [] tree;
        public int depth;

        public BIT(int N);

        public long _bitsum(int i);

        public long bitsum(int l, int r);

        public long bitsum(int l);

        public void bitadd(int i, int x);

        }

    public static boolean check(int X);

    public static void main ( String [ ] args );

}

Step 4: Complete translation of the above Python code to Java code by considering identified classes, functions and Global variables declarations in Step 3. Consider above sample input and output format information to handle input and output related code properly in the generated Java code. While generating the Java code, keep track of required library imports to be added in the Java Code. Make sure your generated code is syntactically correct-
2. Java Code:
import java.util.*;
import java.io.*;
import java.util.stream.Collectors;

public class Main{
    static int n;
    static long m;
    static int[] a;
    static Map<Integer, Integer> d;
    static List<Integer> _a;
    static int[] t;

    public static class BIT {
        public int size;
        public long [] tree;
        public int depth;

        public BIT(int N) {
            this.size = N;
            this.tree = new long[N + 1];
            this.depth = Integer.toBinaryString(N).length();
        }

        public long _bitsum(int i) {
            long ret = 0;
            while (i > 0) {
                ret += this.tree[i];
                i ^= i & -i;
            }
            return ret;
        }

        public long bitsum(int l, int r) {
            return this._bitsum(r) - this._bitsum(l);
        }

        public long bitsum(int l) {
            return this._bitsum(l);
        }

        public void bitadd(int i, int x) {
            i += 1;
            while (i <= this.size) {
                this.tree[i] += x;
                i += i & -i;
            }
            return;
        }
    }


    public static boolean check(int X) {
        int[] b = new int[n + 1];
        b[0] = 0;
        for (int i = 0; i < n; i++) {
            b[i + 1] = (a[i] >= X ? 2 : 0) - 1;
        }
        for (int i = 0; i < n; i++) {
            b[i + 1]+= b[i];
        }
        int c = Arrays.stream(b).min().getAsInt();
        for (int i = 0; i < b.length; i++) {
            b[i] -= c;
        }
        BIT bit = new BIT(Arrays.stream(b).max().getAsInt() + 2);
        long ans = 0;
        for (int x : b) {
            ans += (long) bit.bitsum(x + 1);
            bit.bitadd(x, 1);
        }
        return ans >= m;
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        n = scanner.nextInt();
        m = (long) n * (n + 1) / 4;
        a = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = scanner.nextInt();
        }
        d = new HashMap<>();
        Set<Integer> set = new HashSet<>();
        for (int x : a) {
            set.add(x);
        }
        set.add(0);
        _a = new ArrayList<>(set);
        Collections.sort(_a);
        for (int i = 0; i < _a.size(); i++) {
            d.put(_a.get(i), i);
        }
        for (int i = 0; i < a.length; i++) {
            a[i] = d.get(a[i]);
        }

        t = new int[]{ _a.size(), 0 };

        while (t[0] - t[1] > 1) {
            int mid = (t[0] + t[1]) / 2;
            if (check(mid)) {
                t[1] = mid;
            } else {
                t[0] = mid;
            }
        }
        System.out.println(_a.get(t[1]));
    }
}

###$###
3. Python Code:
'''
        prompt+=code
    if prompt_type=="autocot2d_p2j_v4":#imp formulation with type -final_v3
        prompt='''###$###
1. Python Code:
l = [ ]
limit = 10000000000
def gen ( number , four , seven ) :
    global debug
    debug = True
    status = "bad"
    if ( number > limit ) :
        return
    if ( number > 0 and four == seven ) :
        l.append ( number )
    gen ( number * 10 + 4 , four + 1 , seven )
    gen ( number * 10 + 7 , four , seven + 1 )

debug= False
status = "good"
def main ( ) :
    gen ( 0 , 0 , 0 )
    l.sort ( )
    n = int ( input ( ) )
    ans = 0
    for val in l :
        if ( val >= n ) :
            ans = val
            break
    print ( ans )

1. Sample Input:
9

1. Expected Output:
47


1. Steps: Let's think step by step.

Step 1: First of all identify global variables. Global variables are variables that are in global scope and not inside scope of any function and class present in the above Python Code-
Global variables: l, limit, debug, status

Step 2: Identify declarations of classes and functions present in the above Python Code-
Identified declarations of classes and functions in the Python code:
def gen ( number , four , seven ) :

def main ( ) :

Step 3: By considering above Step 1 and Step 2, identify declarations of these classes, functions and Global variables should be presented in the corresponding Java Code-
Identified declarations of classes, functions and Global variables in the Java code:
public class Main {
    static ArrayList<Long> l;
    static long limit;
    static boolean debug;
    static String status;
    
    public static void gen(long number, int four, int seven);

    public static void main ( String [ ] args );
}

Step 4: Complete translation of the above Python code to Java code by filling definitions of identified declarations of classes, functions and Global variables in Step 3. Consider above sample input and output format information to handle input and output related code properly in the generated Java code. While generating the Java code, keep track of required library imports to be added in the Java Code. Make sure your generated code is syntactically correct-
1. Java Code:
import java.util.*;

public class Main{
    static ArrayList<Long> l;
    static long limit;
    static boolean debug;
    static String status;

    public static void gen(long number, int four, int seven) {
        Main.debug = True;
        String status = "bad";
        if (number > limit) return;
        if (number > 0 && four == seven) {
            l.add(number);
        }
        gen(number * 10 + 4, four + 1, seven);
        gen(number * 10 + 7, four, seven + 1);
    }

    public static void main(String[] args) {
        l = new ArrayList<>();
        limit = 10000000000L;
        debug= False;
        status = "good";

        gen(0, 0, 0);
        Collections.sort(l);
        Scanner sc = new Scanner(System.in);
        long n = sc.nextLong();
        long ans = 0;
        for (long val : l) {
            if (val >= n) {
                ans = val;
                break;
            }
        }
        System.out.println(ans);            
    }  
}

###$###
2. Python Code:
class BIT :
    def __init__ ( self , N ) :
        self.size = N
        self.tree = [ 0 ] * ( N + 1 )
        self.depth = n.bit_length ( )
    def _bitsum ( self , i ) :
        ret = 0
        while i :
            ret += self.tree [ i ]
            i ^= i & - i
        return ret
    def bitsum ( self , l , r = None ) :
        if r is None :
            return self._bitsum ( l )
        else :
            return self._bitsum ( r ) - self._bitsum ( l )
    def bitadd ( self , i , x ) :
        i += 1
        while i <= self.size :
            self.tree [ i ] += x
            i += i & - i
        return
n = int ( input ( ) )
m = n * ( n + 1 ) // 4
a = list ( map ( int , input ( ).split ( ) ) )
d = dict ( )
_a = sorted ( set ( a + [ 0 ] ) )
for i , x in enumerate ( _a ) :
    d [ x ] = i
a = [ d [ x ] for x in a ]
def check ( X ) :
    b = [ 0 ] + [ ( y >= X ) * 2 - 1 for y in a ]
    for i in range ( n ) :
        b [ i + 1 ] += b [ i ]
    c = min ( b )
    b = [ x - c for x in b ]
    bit = BIT ( max ( b ) + 2 )
    ans = 0
    for x in b :
        ans += bit.bitsum ( x + 1 )
        bit.bitadd ( x , 1 )
    return ans >= m
t = [ len ( _a ) , 0 ]
while t [ 0 ] - t [ 1 ] > 1 :
    mid = ( t [ 0 ] + t [ 1 ] ) // 2
    t [ check ( mid ) ] = mid
print ( _a [ t [ 1 ] ] )

1. Sample Input:
1
1

1. Expected Output:
1


1. Steps: Let's think step by step.

Step 1: First of all identify global variables. Global variables are variables that are in global scope and not inside scope of any function and class present in the above Python Code-
Global variables: n, m, a, d, _a, t

Step 2: Identify declarations of classes and functions present in the above Python Code-
Identified declarations of classes and functions in the Python code:
class BIT :
    def __init__ ( self , N ) :
        self.size = N
        self.tree = [ 0 ] * ( N + 1 )
        self.depth = N.bit_length ( )
        
    def _bitsum ( self , i ) :

    def bitsum ( self , l , r = None ) :

    def bitadd ( self , i , x ) :

def check ( X ) :

Step 3: By considering above Step 1 and Step 2, identify declarations of these classes, functions and Global variables should be presented in the corresponding Java Code-
Identified declarations of classes, functions and Global variables in the Java code:
public class Main {
    static int n;
    static long m;
    static List<Integer> a;
    static Map<Integer, Integer> d;
    static List<Integer> _a;
    static int[] t;

    public static class BIT {
        public int size;
        public long [] tree;
        public int depth;

        public BIT(int N);

        public long _bitsum(int i);

        public long bitsum(int l, int r);
    
        public long bitsum(int l);
        
        public void bitadd(int i, int x);
        
        }
        
    public static boolean check(int X);

}

Step 4: Complete translation of the above Python code to Java code by filling definitions of identified declarations of classes, functions and Global variables in Step 3. Consider above sample input and output format information to handle input and output related code properly in the generated Java code. While generating the Java code, keep track of required library imports to be added in the Java Code. Make sure your generated code is syntactically correct-
2. Java Code:
import java.util.*;
import java.io.*;
import java.util.stream.Collectors;

public class Main{
    static int n;
    static long m;
    static int[] a;
    static Map<Integer, Integer> d;
    static List<Integer> _a;
    static int[] t;

    public static class BIT {
        public int size;
        public long [] tree;
        public int depth;

        public BIT(int N) {
            this.size = N;
            this.tree = new long[N + 1];
            this.depth = Integer.toBinaryString(N).length();
        }

        public long _bitsum(int i) {
            long ret = 0;
            while (i > 0) {
                ret += this.tree[i];
                i ^= i & -i;
            }
            return ret;
        }

        public long bitsum(int l, int r) {
            return this._bitsum(r) - this._bitsum(l);
        }

        public long bitsum(int l) {
            return this._bitsum(l);
        }

        public void bitadd(int i, int x) {
            i += 1;
            while (i <= this.size) {
                this.tree[i] += x;
                i += i & -i;
            }
            return;
        }
    }


    public static boolean check(int X) {
        int[] b = new int[n + 1];
        b[0] = 0;
        for (int i = 0; i < n; i++) {
            b[i + 1] = (a[i] >= X ? 2 : 0) - 1;
        }
        for (int i = 0; i < n; i++) {
            b[i + 1]+= b[i];
        }
        int c = Arrays.stream(b).min().getAsInt();
        for (int i = 0; i < b.length; i++) {
            b[i] -= c;
        }
        BIT bit = new BIT(Arrays.stream(b).max().getAsInt() + 2);
        long ans = 0;
        for (int x : b) {
            ans += (long) bit.bitsum(x + 1);
            bit.bitadd(x, 1);
        }
        return ans >= m;
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        n = scanner.nextInt();
        m = (long) n * (n + 1) / 4;
        a = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = scanner.nextInt();
        }
        d = new HashMap<>();
        Set<Integer> set = new HashSet<>();
        for (int x : a) {
            set.add(x);
        }
        set.add(0);
        _a = new ArrayList<>(set);
        Collections.sort(_a);
        for (int i = 0; i < _a.size(); i++) {
            d.put(_a.get(i), i);
        }
        for (int i = 0; i < a.length; i++) {
            a[i] = d.get(a[i]);
        }

        t = new int[]{ _a.size(), 0 };

        while (t[0] - t[1] > 1) {
            int mid = (t[0] + t[1]) / 2;
            if (check(mid)) {
                t[1] = mid;
            } else {
                t[0] = mid;
            }
        }
        System.out.println(_a.get(t[1]));
    }
}

###$###
3. Python Code:
'''
        prompt+=code

    if prompt_type=="autocot2d_j2p" and source_lang=="Java":#v2-j2p
        prompt='''###$###
1. Java Code:
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

public class Main {
    public static void main(String[] args) {
        FastScanner input = new FastScanner();
        int n = input.nextInt();
        HashMap<Integer, Integer> map = new HashMap<>();
        
        for (int i = 0; i < n; i++) {
            int val = input.nextInt();
            map.put(val, map.getOrDefault(val, 0) + 1);
        }
        
        int max = Integer.MIN_VALUE;
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            Integer value = entry.getValue();
            max = Math.max(max, value);
        }
        
        System.out.println(max + " " + map.size());
    }

    static class FastScanner {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        StringTokenizer st = new StringTokenizer("");

        String next() {
            while (!st.hasMoreTokens()) {
                try {
                    st = new StringTokenizer(br.readLine());
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            return st.nextToken();
        }

        int nextInt() {
            return Integer.parseInt(next());
        }

        long nextLong() {
            return Long.parseLong(next());
        }

        double nextDouble() {
            return Double.parseDouble(next());
        }

        String nextLine() throws IOException {
            return br.readLine();
        }
    }
}

1. Sample Input:
1
1

1. Expected Output:
1 1


1. Steps: Let's think step by step.

Step 1: First of all identify classes and functions declarations present in the above Java Code-
Identified classes and functions declarations in the Java code:
public class Main { 
    public static void main ( String [ ] args );
    static class FastScanner { 
        String next ( );
        int nextInt ( );
        long nextLong ( );
        double nextDouble ( );
        String nextLine ( ) throws IOException; 
    }
}

Step 2: Considering Step 1, classes and functions declarations should be presented in the corresponding Python Code-
Identified classes and functions declarations in the Python code:
Class Main:
    def main(self):

    class FastScanner:
        def __init__(self):

        def next(self):

        def nextInt(self):

        def nextLong(self):

        def nextDouble(self):

        def nextLine(self):


Step 3: Complete translation of the above Java code to Python code by considering identified classes and functions declarations in Step 2 only. Consider above sample input and output format information to handle input and output related code properly in the generated Python code. While generating the Python code, keep track of required library imports to be added in the Python Code. Make sure your generated code is syntactically correct-
1. Python Code:
import sys

class Main:
    def main(self):
        input = self.FastScanner()
        n = input.nextInt()
        map = {}
        for i in range(n):
            val = input.nextInt()
            map[val] = map.get(val, 0) + 1
        max = -1
        for entry in map.items():
            value = entry[1]
            max = max if max > value else value
        print(max, len(map))

    class FastScanner:
        def __init__(self):
            self.buf = sys.stdin.readline
            self.tokens = None

        def next(self):
            while self.tokens is None or len(self.tokens) == 0:
                self.tokens = self.buf().split()
            if (self.tokens is None or len(self.tokens) == 0):
                return ''
            token = self.tokens[0]
            self.tokens = self.tokens[1:]
            return token

        def nextInt(self):
            return int(self.next())

        def nextLong(self):
            return int(self.next())

        def nextDouble(self):
            return float(self.next)

        def nextLine(self):
            return self.buf()

if __name__ == "__main__":
    obj = Main()
    obj.main()


###$###
2. Java Code:
import java.util.Scanner ;

public class Main {
    public static void main ( String [ ] args ) {
        Scanner in_ = new Scanner ( System.in ) ;
        int F = in_.nextInt ( ) ;
        int T = in_.nextInt ( ) ;
        int S = in_.nextInt ( ) ;
        int q = in_.nextInt ( ) ;
        long previous = S ;
        int answer = 0 ; 
        while ( previous < T ) {
            answer ++ ; previous *= q ; 
        }
        answer*=F;
        System.out.println ( answer ) ; 
} }

2. Sample Input:
4
5 2 2

2. Expected Output:
8


2. Steps: Let's think step by step.

Step 1: First of all identify classes and functions declarations present in the above Java Code-
Identified classes and functions declarations in the Java code:
public class Main {
    public static void main ( String [ ] args );
}

Step 2: Considering Step 1, classes and functions declarations should be presented in the corresponding Python Code-
Identified classes and functions declarations in the Python code:
Class Main:
    def main(self):


Step 3: Complete translation of the above Java code to Python code by considering identified classes and functions declarations in Step 2 only. Consider above sample input and output format information to handle input and output related code properly in the generated Python code. While generating the Python code, keep track of required library imports to be added in the Python Code. Make sure your generated code is syntactically correct-
2. Python Code:
class Main:
    def main(self):
        F=int(input().strip())
        T, S, q = map ( int, input ( ).split ( ) )
        previous = S
        answer = 0
        while ( previous < T ):
            answer+=1
            previous *= q
        answer*=F
        print( answer )

if __name__ == "__main__":
    obj = Main()
    obj.main()

###$###
3. Java Code:
'''
        prompt+=code
    if prompt_type=="autocot2d" and source_lang=="Java":#v1-j2p
        prompt='''
###$###
1. Java Code:
import java.io.BufferedReader ; 
import java.io.IOException ; 
import java.io.InputStreamReader ; 
import java.util.* ; 

public class Main { 
public static void main ( String [ ] args ) { 
FastScanner input = new FastScanner ( ) ; 
int n = input.nextInt ( ) ; 
HashMap < Integer , Integer > map = new HashMap < > ( ) ; 
for ( int i = 0 ; i < n ; i ++ ) { 
int val = input.nextInt ( ) ; 
map.put ( val , map.getOrDefault ( val , 0 ) + 1 ) ;
} 
int max = Integer.MIN_VALUE ; 
for ( Map.Entry < Integer , Integer > entry : map.entrySet ( ) ) { 
Integer value = entry.getValue ( ) ;
max = Math.max ( max , value ) ;
} 
System.out.println ( max + " " + map.size ( ) ) ; 
}
static class FastScanner { 
BufferedReader br = new BufferedReader ( new InputStreamReader ( System.in ) ) ;
StringTokenizer st = new StringTokenizer ( "" ) ;

String next ( ) { while ( ! st.hasMoreTokens ( ) ) {
try { st = new StringTokenizer ( br.readLine ( ) ) ;
} catch ( IOException e ) {
e.printStackTrace ( ) ; 
} } 
return st.nextToken ( ) ;
} 
int nextInt ( ) {
return Integer.parseInt ( next ( ) ) ;
} 
long nextLong ( ) { 
return Long.parseLong ( next ( ) ) ; 
} 
double nextDouble ( ) {
return Double.parseDouble ( next ( ) ) ; 
} 
String nextLine ( ) throws IOException {
return br.readLine ( ) ; 
} } }

1. Sample Input:
1
1

1. Expected Output:
1 1


1. Steps: Let's think step by step.

Step 1: First of all identify classes and functions declarations present in the above Java Code-
Identified classes and functions declarations in the Java code:
public class Main { 
public static void main ( String [ ] args );
static class FastScanner { 
String next ( );
int nextInt ( );
long nextLong ( );
double nextDouble ( );
String nextLine ( ) throws IOException; 
}
}

Step 2: Considering Step 1, classes and functions declarations should be presented in the corresponding Python Code-
Identified classes and functions declarations in the Python code:
Class Main:
    def main(self):

    class FastScanner:
        def __init__(self):

        def has_next(self):

        def next_int(self):

        def next_str(self):


Step 3: Complete translation of the above Java code to Python code by considering identified classes and functions declarations in Step 2. Consider above sample input and output format information to handle input and output related code properly in the generated Python code. While generating the Python code, keep track of required library imports to be added in the Python Code. Make sure your generated code is syntactically correct-
1. Python Code:
import sys

class Main:
    def main(self):
        input = self.FastScanner()
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

if __name__ == "__main__":
    obj = Main()
    obj.main()

###$###
2. Java Code:
import java.util.Scanner ;

public class Main {
    public static void main ( String [ ] args ) {
        Scanner in_ = new Scanner ( System.in ) ;
        int F = in_.nextInt ( ) ;
        int T = in_.nextInt ( ) ;
        int S = in_.nextInt ( ) ;
        int q = in_.nextInt ( ) ;
        long previous = S ;
        int answer = 0 ; 
        while ( previous < T ) {
            answer ++ ; previous *= q ; 
        }
        answer*=F;
        System.out.println ( answer ) ; 
} }

2. Sample Input:
4
5 2 2

2. Expected Output:
8


2. Steps: Let's think step by step.

Step 1: First of all identify classes and functions declarations present in the above Java Code-
Identified classes and functions declarations in the Java code:
public class Main {
    public static void main ( String [ ] args );
}

Step 2: Considering Step 1, classes and functions declarations should be presented in the corresponding Python Code-
Identified classes and functions declarations in the Python code:
Class Main:
    def main(self):


Step 3: Complete translation of the above Java code to Python code by considering identified classes and functions declarations in Step 2. Consider above sample input and output format information to handle input and output related code properly in the generated Python code. While generating the Python code, keep track of required library imports to be added in the Python Code. Make sure your generated code is syntactically correct-
2. Python Code:
class Main:
    def main(self):
        F=int(input().strip())
        T , S , q = map ( int , input ( ).split ( ) )
        previous = S
        answer = 0
        while ( previous < T ):
            answer+=1
            previous *= q
        answer*=F
        print( answer )

if __name__ == "__main__":
    obj = Main()
    obj.main()

###$###
3. Java Code:
'''
        prompt+=code
    if prompt_type=="autocot1" and source_lang=="Java":
        prompt='''
###$###
1. Whole Java Code:
import java.io.BufferedReader ; 
import java.io.IOException ; 
import java.io.InputStreamReader ; 
import java.util.* ; 

public class Main { 
public static void main ( String [ ] args ) { 
FastScanner input = new FastScanner ( ) ; 
int n = input.nextInt ( ) ; 
HashMap < Integer, Integer > map = new HashMap < > ( ) ; 
for ( int i = 0 ; i < n ; i ++ ) { 
int val = input.nextInt ( ) ; 
map.put ( val, map.getOrDefault ( val, 0 ) + 1 ) ;
} 
int max = Integer.MIN_VALUE ; 
for ( Map.Entry < Integer, Integer > entry : map.entrySet ( ) ) { 
Integer value = entry.getValue ( ) ;
max = Math.max ( max, value ) ;
} 
System.out.println ( max + "" + map.size ( ) ) ; 
}
static class FastScanner { 
BufferedReader br = new BufferedReader ( new InputStreamReader ( System.in ) ) ;
StringTokenizer st = new StringTokenizer ( "" ) ;

String next ( ) { while (! st.hasMoreTokens ( ) ) {
try { st = new StringTokenizer ( br.readLine ( ) ) ;
} catch ( IOException e ) {
e.printStackTrace ( ) ; 
} } 
return st.nextToken ( ) ;
} 
int nextInt ( ) {
return Integer.parseInt ( next ( ) ) ;
} 
long nextLong ( ) { 
return Long.parseLong ( next ( ) ) ; 
} 
double nextDouble ( ) {
return Double.parseDouble ( next ( ) ) ; 
} 
String nextLine ( ) throws IOException {
return br.readLine ( ) ; 
} } }

1. Sample Input:
1
1

2. Expected Output:
1 1


1. Steps: Let's think step by step. 
Firstly find classes defined in the Java code that are- FastScanner, Main. 
Let's translate all classes definition and its functions step by step. Here is order in which steps to be followed- First translate all classes instead of "Main" class, then translate  "Main" class and then "library imports" step.

Let's translate FastScanner class. Functions present in FastScanner class- next, nextInt, nextLong, nextDouble, nextLine. Translation of FastScanner class and its functions from Java to Python-
Step 1: Translation of "FastScanner" class and its functions -
Python Code:
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

Let's translate Main class. Functions present in Main class- main. Consider above sample input and output format information  while translating "main" function to handle input related code properly. Translation of Main class and its functions from Java to Python-
Step 2: Translation of "Main" class and its functions -
Python Code:
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

Considering all steps, add required library imports in Python Code-
Step 3: Library imports -
Python Code:
import sys

Finally Combined all steps- Step 1, Step 2, Step 3 to get syntactically correct whole Python Code-
1. Whole Python Code:
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

###$###
2. Whole Java Code:
import java.util.Scanner ;

public class Main {
    public static void main ( String [ ] args ) {
        Scanner in_ = new Scanner ( System.in ) ;
        int F = in_.nextInt ( ) ;
        int T = in_.nextInt ( ) ;
        int S = in_.nextInt ( ) ;
        int q = in_.nextInt ( ) ;
        long previous = S ;
        int answer = 0 ; 
        while ( previous < T ) {
            answer ++ ; previous *= q ; 
        }
        answer*=F;
        System.out.println ( answer ) ; 
} }

2. Sample Input:
4
5 2 2

2. Expected Output:
8


2. Steps: Let's think step by step. 
Firstly find classes defined in the Java code that are- Main. 
Let's translate all classes definition and its functions step by step. Here is order in which steps to be followed- First translate all classes instead of "Main" class, then translate  "Main" class and then "library imports" step.

Let's translate Main class. Functions present in Main class- main. Consider above sample input and output format information  while translating "main" function to handle input related code properly. Translation of Main class and its functions from Java to Python-
Step 1: Translation of "Main" class and its functions -
Python Code:
def main():
    F=int(input().strip())
    T, S, q = map ( int, input ( ).split ( ) )
    previous = S
    answer = 0
    while ( previous < T ):
        answer+=1
        previous *= q
    answer*=F
    print( answer )

Considering all steps, add required library imports in Python Code-
Step 2: Library imports -
Python Code:
# No library import required in Python Code

Finally Combined all steps- Step 1, Step 2 to get syntactically correct whole Python Code.
2. Whole Python Code:
def main():
    F=int(input().strip())
    T, S, q = map ( int, input ( ).split ( ) )
    previous = S
    answer = 0
    while ( previous < T ):
        answer+=1
        previous *= q
    answer*=F
    print( answer )

if __name__ == "__main__":
    main()

###$###
3. Whole Java Code:
'''
        prompt+=code
    if prompt_type == 'gpt' or prompt_type == 'gemini':
        prompt = f'You are a code translation expert. Translate the {source_lang} code below to {target_lang}\n\n{source_lang}\n{code}\n\n{target_lang}\n'

    if prompt_type == 'claude':
        prompt = f'\n```Translate the {source_lang} code below to {target_lang}\n\n{source_lang}\n{code}\n\n{target_lang}\n```\n'

    if prompt_type == 'codellama':
        prompt = f'<s>[INST] <<SYS>> You are a code translation expert. <</SYS>>\n\nTranslate the {source_lang} code below to {target_lang} and end with comment \"<END-OF-CODE>\"\n\n[/INST]\n\n{source_lang}:\n{code}\n\n{target_lang}:\n'

    if prompt_type == 'octocoder':
        prompt = f'Question: You are a code translation expert. Translate the {source_lang} code below to {target_lang} and end with comment \"<END-OF-CODE>\"\n\n{source_lang}\n{code}\n\nAnswer:\n {target_lang}\n'

    if prompt_type == 'dolphin' or prompt_type == 'mistral-hermes':
        prompt = f"""<|im_start|>system
                You are a code translation expert.<|im_end|>
                <|im_start|>user
                Can you translate the following {source_lang} code into {target_lang} and end with comment \"<END-OF-CODE>\"?
                ```{source_lang}
                {code}
                ```
                <|im_end|>
                <|im_start|>assistant
                ```{target_lan}
                """

    if prompt_type == 'starcoder':
        prompt = f"<fim_prefix>{source_lang}:\n{code}\n{target_lang}:\n<fim_suffix><fim_middle>"
                

    if prompt_type == 'solar':
        prompt = f"""<s> ### User:
        Can you translate the {source_lang} code below to {target_lang} and end with comment \"<END-OF-CODE>\"?
        ```{source_lang}
        {code}
        ```

        ### Assistant:
        Sure!
        ```{target_lang}
        """       

    if prompt_type == 'wizardcoder':
        prompt = f"""You are a code translation expert. Below is an instruction that describes a code translation task. Write a response that appropriately completes the request.

        ### Instruction:
        Write {target_lang} code that translates the following {source_lang} code and end with comment\"<END-OF-CODE>\":
        {code}

        ### Response:"""    

    if prompt_type == "deepseek":
        prompt = f'''You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
        ### Instruction:
        Translate the following {source_lang} code to {target_lang}.\n\n{source_lang}\n{code}

        ### Response:
        '''

    if prompt_type == "phi":   
         prompt = f'Instruct: You are a code translation expert. Translate the {source_lang} code below to {target_lang} and end with comment \"<END-OF-CODE>\"\n\n{source_lang}\n{code}\n\nOutput:\n {target_lang}\n'   

    if prompt_type == 'magic':
        
        prompt = f"""You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

        @@ Instruction
        You are a code translation expert. Translate the {source_lang} code below to {target_lang} and end with comment \"<END-OF-CODE>\"\n\n{source_lang}\n{code}\n'

        @@ Response
        {target_lang}
        """     

    if prompt_type == 'vicuna':
        prompt = f"""### System Prompt
                     You are a code translation expert.

                    ### User Message
                    Translate the {source_lang} code below to {target_lang} and end with comment \"<END-OF-CODE>\"\n\n{source_lang}\n{code}\n

                    ### Assistant
                   """

    return prompt


# Adopted from https://github.com/huggingface/transformers/pull/14897
class EndOfFunctionCriteria(StoppingCriteria):
    def __init__(self, start_length, eos, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_length = start_length
        self.eos = eos
        self.tokenizer = tokenizer
        self.end_length = {}

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length :]
        )
        done = []
        for index, decoded_generation in enumerate(decoded_generations):
            finished = any(
                [stop_string in decoded_generation for stop_string in self.eos]
            )
            if (
                finished and index not in self.end_length
            ):  # ensures first time we see it
                for stop_string in self.eos:
                    if stop_string in decoded_generation:
                        self.end_length[index] = len(
                            input_ids[
                                index,  # get length of actual generation
                                self.start_length : -len(
                                    self.tokenizer.encode(
                                        stop_string,
                                        add_special_tokens=False,
                                        return_tensors="pt",
                                    )[0]
                                ),
                            ]
                        )
            done.append(finished)
        return all(done)


class DecoderBase(ABC):
    def __init__(
        self,
        name: str,
        batch_size: int = 1,
        temperature: float = 0.8,
        conversational: bool = False,
        tensor_parallel_size: int = 1        
    ) -> None:
        print("Initializing a decoder model: {} ...".format(name))
        self.name = name
        self.batch_size = batch_size
        self.temperature = temperature
        self.eos = EOS
        self.skip_special_tokens = False
        self.conversational = conversational
        self.tensor_parallel_size = tensor_parallel_size

    @abstractmethod
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200, max_length: int = 1024
    ) -> List[str]:
        pass


    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

# NOTE: in order to use Gemini, the GEMINI_KEY environment variable must be set 
class GeminiDecoder(DecoderBase, ABC):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
   
        genai.configure(api_key=os.environ.get('GEMINI_KEY'))

        self.model = genai.GenerativeModel(name)

    def codegen(self, prompt: str, do_sample: bool = True, num_samples: int = 200, max_length: int = 1024) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"

        batch_size = min(self.batch_size, num_samples)
        if not do_sample:
            assert batch_size == 1, "Sampling only supports batch size of 1"

        outputs = []
        for _ in range(batch_size):
            try:
                response = self.model.generate_content(prompt, 
                                                    generation_config=genai.types.GenerationConfig(
                                                        # Only one candidate for now.
                                                        candidate_count=num_samples,
                                                        max_output_tokens=max_length,
                                                        temperature=self.temperature)
                                                        )
                outputs.append(response.text)                                        
            except:  
                outputs.append('GEMINI API ERROR')    
            

        return outputs    


# NOTE: in order to use Claude, the ANTHROPIC_KEY environment variable must be set 
class AnthropicDecoder(DecoderBase, ABC):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_KEY"))


class AnthropicMessageDecoder(AnthropicDecoder):
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200, max_length: int = 1024
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"

        batch_size = min(self.batch_size, num_samples)
        if not do_sample:
            assert batch_size == 1, "Sampling only supports batch size of 1"

        outputs = []
        for _ in range(batch_size):
            message = anthropic_request.make_auto_request(
                client=self.client,
                model=self.name,
                system="You are a code translation expert.",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_length,
                temperature=self.temperature,
                stop_sequences=["\n```\n"],
            )
            outputs.append(message.content[0].text)

        return outputs

class VLlmDecoder(DecoderBase):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)

        kwargs = {"tensor_parallel_size": self.tensor_parallel_size}
        if "CodeLlama" in name:
            kwargs["dtype"] = "bfloat16"
        elif "CodeBooga" in name:
            kwargs["dtype"] = "float16"
        elif "WizardCoder" in name:
            kwargs["dtype"] = "float16"
        elif "deepseek" in name:
            kwargs["dtype"] = "bfloat16"
        elif "mixtral" in name.lower():
            kwargs["dtype"] = "bfloat16"
        elif "solar" in name:
            kwargs["dtype"] = "float16"
        elif "mistral" in name.lower():
            kwargs["dtype"] = "bfloat16"
        elif "phi" in name.lower():
            kwargs["dtype"] = "float16"
            kwargs["trust_remote_code"] = True

              
        self.path = name
   
        self.llm = LLM(model=self.path, **kwargs)

        self.context_window_length = self.llm.get_tokenizer().model_max_length
        if self.context_window_length > 30000:

            # find config file 
            p = [x for x in os.listdir(cache_dir) if x.find(name.split('/')[-1])>0]
            p = [x for x in p if os.path.isdir(f'{cache_dir}/{x}')][0]
            self.path = f'{cache_dir}/{p}'

            config_path = None
            for path in glob.glob(f'{self.path}/**/config.json', recursive=True):
                config_path = path

            if config_path:
                with open(config_path) as fin:
                    config_data = json.load(fin)

                if 'n_positions' in config_data:
                    self.context_window_length = config_data['n_positions']
                elif 'max_position_embeddings' in config_data:
                    self.context_window_length = config_data['max_position_embeddings']
                else:
                    print('Model has unclear context_window_length, setting to 1024')
                    self.context_window_length = 1024
            else:
                print('Model has unclear context_window_length, setting to 1024')
                self.context_window_length = 1024   

  


    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200, max_length: int = 1024
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"
        batch_size = min(self.batch_size, num_samples)

        vllm_outputs = self.llm.generate(
            [prompt] * batch_size,
            SamplingParams(
                temperature=self.temperature,
                max_tokens=min(max_length, self.context_window_length),
                top_p=0.95 if do_sample else 1.0,
                stop=self.eos,
            ),
            use_tqdm=False,
        )

        gen_strs = [x.outputs[0].text.replace("\t", "    ") for x in vllm_outputs]
        return gen_strs


# chatml format
class ChatML(VLlmDecoder):
    def __init__(self, name: str, **kwargs) -> None:
        kwargs["conversational"] = True
        super().__init__(name, **kwargs)
        self.eos += ["\n```"]


    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200, max_length: int = 1024
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"

        return VLlmDecoder.codegen(self, prompt, do_sample, num_samples, max_length)


class HFTorchDecoder(DecoderBase):
    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = {
            "trust_remote_code": name
            in {
                "bigcode/santacoder",
                "Salesforce/codegen2-1B",
                "Salesforce/codegen2-3_7B",
                "Salesforce/codegen2-7B",
                "Salesforce/codegen2-16B",
                "deepseek-ai/deepseek-coder-1.3b-base",
                "deepseek-ai/deepseek-coder-6.7b-base",
                "deepseek-ai/deepseek-coder-33b-base",
                "deepseek-ai/deepseek-coder-1.3b-instruct",
                "deepseek-ai/deepseek-coder-6.7b-instruct",
                "deepseek-ai/deepseek-coder-33b-instruct"
            }
        }

        if "codegen-" in name:  # use fp16 for codegen models
            kwargs["torch_dtype"] = torch.float16
        if "codegen2-" in name:  # avoid warning of trust remote code
            kwargs["revision"] = "main"
            if "16b" in name.lower():
                kwargs["device_map"] = "auto"
        if "starcoder" in name:
            kwargs["torch_dtype"] = torch.bfloat16
            kwargs["device_map"] = "auto"
            model_path_starcoder="/scratch/scai/mtech/aib222684/starcoder"
            cache_dir_starcoder="/scratch/cse/dual/cs5190439/MTP1/starcoder_data"
            os.makedirs(cache_dir_starcoder, exist_ok=True)
            # kwargs["cache_dir"]=cache_dir_starcoder
            self.tokenizer = AutoTokenizer.from_pretrained(model_path_starcoder, use_auth_token=True, cache_dir=cache_dir_starcoder)
            self.model = AutoModelForCausalLM.from_pretrained(
        		model_path_starcoder,
        		use_auth_token=True,cache_dir=cache_dir_starcoder,**kwargs)
        if "granite" in name:
            # from accelerate import infer_auto_device_map
            # device_map_ = infer_auto_device_map(my_model, max_memory={0: "10GiB", 1: "10GiB", "cpu": "30GiB"})
            if ("20b" in name):
                kwargs["torch_dtype"] = torch.bfloat16
            # not in 8b possible
            kwargs["device_map"] ="auto" #"auto"
            model_path=f"/scratch/scai/mtech/aib222684/{name}"
            cache_dir_granite=f"/scratch/cse/dual/cs5190439/MTP1/{name}_data"
            os.makedirs(cache_dir_granite, exist_ok=True)
            # kwargs["cache_dir"]=cache_dir_starcoder
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=True, cache_dir=cache_dir_granite)
            self.model = AutoModelForCausalLM.from_pretrained(
        		model_path,use_auth_token=True,cache_dir=cache_dir_granite,**kwargs)
            self.model.eval()
        	
        if "CodeLlama" in name:
            # if "34b" in name.lower():
            #     kwargs["device_map"] = "auto"
            kwargs["torch_dtype"] = torch.bfloat16
            kwargs["device_map"] ="auto" #"auto"
            self.skip_special_tokens = True
            model_path=f"/scratch/scai/mtech/aib222684/{name}"
            cache_dir_codellama=f"/scratch/cse/dual/cs5190439/MTP1/{name}_data"
            os.makedirs(cache_dir_codellama, exist_ok=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=True, cache_dir=cache_dir_codellama)
            self.model = AutoModelForCausalLM.from_pretrained( model_path,use_auth_token=True,cache_dir=cache_dir_codellama,**kwargs)
            self.model.generation_config.eos_token_id=self.tokenizer.eos_token_id
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        if "CodeBooga" in name:
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto"
            self.skip_special_tokens = True
        if "Mistral-7B-codealpaca-lora" == name:
            kwargs["torch_dtype"] = torch.float16
            self.skip_special_tokens = True
        elif "Mistral" in name or "zephyr-7b-beta" in name:
            kwargs["torch_dtype"] = torch.bfloat16
        if "deepseek" in name:
            kwargs["device_map"] = "auto"
            kwargs["torch_dtype"] = torch.bfloat16
            self.skip_special_tokens = True
        if "/phi" in name:
            kwargs["torch_dtype"] = torch.float16
            kwargs["trust_remote_code"] = True
            self.skip_special_tokens = True

        print(f"{kwargs} = ")

        # self.tokenizer = AutoTokenizer.from_pretrained(name)
        # self.model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        if name in {"StabilityAI/stablelm-base-alpha-7b"}:
            print("Switching to float16 ...")
            self.model = self.model.half()
            self.skip_special_tokens = True

        if kwargs["device_map"] != "auto":
            self.model = self.model.to(self.device)


        self.context_window_length = self.tokenizer.model_max_length
        if self.context_window_length > 1000000:
            if hasattr(self.model.config, 'n_positions'):
                self.context_window_length = self.model.config.n_positions
            elif hasattr(self.model.config, 'max_position_embeddings'):
                self.context_window_length = self.model.config.max_position_embeddings
            else:
                print('Model has unclear context_window_length, setting to 1024')
                self.context_window_length = 4090#1024

    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200, max_length: int = 1024
    ) -> List[str]:
        if self.temperature == 0:
            assert not do_sample
            assert num_samples == 1

        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.device
        )
        scores = StoppingCriteriaList(
            [
                EndOfFunctionCriteria(
                    start_length=len(input_tokens[0]),
                    eos=self.eos,
                    tokenizer=self.tokenizer,
                )
            ]
        )
        kwargs = {}
        if do_sample:
            kwargs["top_p"] = 0.95
            kwargs["temperature"] = self.temperature
            # kwargs["temperature"] = max(self.temperature, 1e-2)

        if ("granite" in self.name or "CodeLlama" in self.name): #like strcoder infill code
            self.context_window_length =max_length #4090

            if len(input_tokens[0]) >= self.context_window_length :
                outputs = []
                for _ in range(num_samples):
                    outputs.append('MODEL MAX LENGTH EXCEEDED')
                return outputs    

        raw_outputs = self.model.generate(
            input_tokens,
            max_new_tokens=min(max_length, self.context_window_length - len(input_tokens[0])),
            stopping_criteria=scores,
            do_sample=do_sample,
            output_scores=True,
            return_dict_in_generate=True,
            num_return_sequences=min(self.batch_size, num_samples),
            pad_token_id=self.tokenizer.eos_token_id,#"CodeLlama" in name
            **kwargs,
        )  # remove warning
        gen_seqs = raw_outputs.sequences[:, len(input_tokens[0]) :]
        gen_strs = self.tokenizer.batch_decode(
            gen_seqs, skip_special_tokens=self.skip_special_tokens
        )
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    # could be multiple eos in outputs, better pick minimum one
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index])
        return outputs


class DeepSeekInstruct(HFTorchDecoder):
    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200, max_length: int = 1024
    ) -> List[str]:
        if self.temperature == 0:
            assert not do_sample
            assert num_samples == 1

        input_tokens = self.tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": f"{prompt}",
                }
            ],
            return_tensors="pt",
        ).to(self.device)
        kwargs = {}
        if do_sample:
            kwargs["top_p"] = 0.95
            kwargs["temperature"] = self.temperature


        raw_outputs = self.model.generate(
            input_tokens,
            max_new_tokens=min(max_length, self.context_window_length - len(input_tokens[0])),
            do_sample=do_sample,
            output_scores=True,
            return_dict_in_generate=True,
            top_k=50,
            num_return_sequences=min(self.batch_size, num_samples),
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=32021,
            **kwargs,
        )  # remove warning
        gen_seqs = raw_outputs.sequences[:, len(input_tokens[0]) :]
        gen_strs = self.tokenizer.batch_decode(
            gen_seqs, skip_special_tokens=self.skip_special_tokens
        )
        return gen_strs

# NOTE: in order to use gpt, the OPENAI_API_KEY environment variable must be set 
class OpenAIChatDecoder(DecoderBase):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.client = openai.OpenAI()

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200, max_length: int = 1024
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"

        batch_size = min(self.batch_size, num_samples)
        assert batch_size <= 20, "Use larger batch size could blow up the memory!"

        # construct prompt
        fmt = "text"

        ret = make_auto_request(
            self.client,
            message=prompt,
            model=self.name,
            max_tokens=self.max_length,
            temperature=self.temperature,
            n=batch_size,
            response_format={"type": fmt},
        )


        outputs = []
        for item in ret.choices:
            content = item.message.content
            # if json serializable
            if fmt == "json_object":
                try:
                    json_data = json.loads(content)
                    if json_data.get("code", None) is not None:
                        outputs.append(prompt + "\n" + json_data["code"])
                        continue

                    print(f"'code' field not found in: {json_data}")
                except Exception as e:
                    print(e)
            outputs.append(content)

        return outputs


class IncoderDecoder(HFTorchDecoder):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.infill_ph = "<|mask:0|>"
        self.extra_end = "<|mask:1|><|mask:0|>"
        self.extra_eos = [
            "<|endofmask|>",
            "<|/ file",
            "</cell>",
            "</text>",
            "</code>",
            "<|",
            "</CODE>",
        ]
        self.eos = self.eos + self.extra_eos

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200, max_length: int = 1024
    ) -> List[str]:
        input = prompt + self.infill_ph + self.extra_end
        input_tokens = self.tokenizer.encode(input, return_tensors="pt").to(self.device)
        scores = StoppingCriteriaList(
            [
                EndOfFunctionCriteria(
                    start_length=len(input_tokens[0]),
                    eos=self.eos,
                    tokenizer=self.tokenizer,
                )
            ]
        )


        raw_outputs = self.model.generate(
            input_tokens,
            max_new_tokens=min(max_length, self.context_window_length - len(input_tokens[0])),
            stopping_criteria=scores,
            do_sample=do_sample,
            top_p=0.95,
            top_k=None,
            temperature=self.temperature,
            num_return_sequences=min(self.batch_size, num_samples),
            output_scores=True,
            return_dict_in_generate=True,
        )
        gen_seqs = raw_outputs.sequences[:, len(input_tokens[0]) :]
        gen_strs = self.tokenizer.batch_decode(
            gen_seqs, skip_special_tokens=self.skip_special_tokens
        )
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index])
        return outputs


class Codegen2Decoder(HFTorchDecoder):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.infill_ph = "<mask_1>"
        # taken from: https://huggingface.co/Salesforce/codegen2-16B
        self.extra_end = "<|endoftext|><sep><mask_1>"
        self.extra_eos = ["<eom>"]
        self.eos = self.eos + self.extra_eos

    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200, max_length: int = 1024
    ) -> List[str]:
        if self.temperature == 0:
            assert not do_sample
            assert num_samples == 1

        input = prompt + self.infill_ph + self.extra_end
        input_tokens = self.tokenizer.encode(input, return_tensors="pt").to(self.device)
        scores = StoppingCriteriaList(
            [
                EndOfFunctionCriteria(
                    start_length=len(input_tokens[0]),
                    eos=self.eos,
                    tokenizer=self.tokenizer,
                )
            ]
        )


        raw_outputs = self.model.generate(
            input_tokens,
            max_new_tokens=min(max_length, self.context_window_length - len(input_tokens[0])),
            stopping_criteria=scores,
            do_sample=do_sample,
            top_p=0.95,
            top_k=None,
            temperature=self.temperature,
            output_scores=True,
            return_dict_in_generate=True,
            num_return_sequences=min(self.batch_size, num_samples),
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen_seqs = raw_outputs.sequences[:, len(input_tokens[0]) :]
        gen_strs = self.tokenizer.batch_decode(
            gen_seqs, skip_special_tokens=self.skip_special_tokens
        )
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index])
        return outputs


class SantaCoder(HFTorchDecoder):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.prefix_token = "<fim-prefix>"
        self.suffix_token = "<fim-suffix>\n<fim-middle>"
        self.extra_eos = ["<|endofmask|>"]
        self.eos = self.eos + self.extra_eos

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200, max_length: int = 1024
    ) -> List[str]:
        if self.temperature == 0:
            assert not do_sample
            assert num_samples == 1

        input = self.prefix_token + prompt + self.suffix_token
        input_tokens = self.tokenizer.encode(input, return_tensors="pt").to(self.device)
        scores = StoppingCriteriaList(
            [
                EndOfFunctionCriteria(
                    start_length=len(input_tokens[0]),
                    eos=self.eos,
                    tokenizer=self.tokenizer,
                )
            ]
        )

        if len(input_tokens[0]) >= self.context_window_length:
            outputs = []
            for _ in range(num_samples):
                outputs.append('MODEL MAX LENGTH EXCEEDED')
            return outputs

        raw_outputs = self.model.generate(
            input_tokens,
            max_new_tokens=min(max_length, int((self.context_window_length - len(input_tokens[0])) * 0.9)),
            stopping_criteria=scores,
            do_sample=do_sample,
            top_p=0.95,
            top_k=None,
            temperature=self.temperature,
            num_return_sequences=min(self.batch_size, num_samples),
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen_seqs = raw_outputs.sequences[:, len(input_tokens[0]) :]
        gen_strs = self.tokenizer.batch_decode(
            gen_seqs,
            skip_special_tokens=self.skip_special_tokens,
        )
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index])
        return outputs


class StarCoderInfill(HFTorchDecoder):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.prefix_token = "<fim_prefix>"
        self.suffix_token = "<fim_suffix><fim_middle>"

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200, max_length: int = 1024
    ) -> List[str]:
        if self.temperature == 0:
            assert not do_sample
            assert num_samples == 1

        input = self.prefix_token + prompt + self.suffix_token
        input_tokens = self.tokenizer.encode(input, return_tensors="pt").to(self.device)
        scores = StoppingCriteriaList(
            [
                EndOfFunctionCriteria(
                    start_length=len(input_tokens[0]),
                    eos=self.eos,
                    tokenizer=self.tokenizer,
                )
            ]
        )

        kwargs = {}
        if do_sample:
            kwargs["top_p"] = 0.95
            # top_k default to 50
            kwargs["temperature"] = max(self.temperature, 1e-2)
        
        self.context_window_length =max_length #4090
 

        if len(input_tokens[0]) >= self.context_window_length :
            outputs = []
            for _ in range(num_samples):
                outputs.append('MODEL MAX LENGTH EXCEEDED')
            return outputs

        raw_outputs = self.model.generate(
            input_tokens,
            max_new_tokens=min(max_length, int((self.context_window_length  - len(input_tokens[0])))),#0.9 round-off error
            stopping_criteria=scores,
            do_sample=do_sample,
            num_return_sequences=min(self.batch_size, num_samples),
            output_scores=True,
            return_dict_in_generate=True,
            repetition_penalty=1.0,
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs #prev only in HFcoder
        )
        gen_seqs = raw_outputs.sequences[:, len(input_tokens[0]) :]
        gen_strs = self.tokenizer.batch_decode(
            gen_seqs,
            skip_special_tokens=self.skip_special_tokens
        )
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index])
        return outputs


class CodeT5P(DecoderBase):
    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert name in {
            "Salesforce/codet5p-2b",
            "Salesforce/codet5p-6b",
            "Salesforce/codet5p-16b",
            "Salesforce/instructcodet5p-16b",
        }
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            name,
            trust_remote_code=True,  # False for 220m and 770m models
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.model.to(self.device)

        self.skip_special_tokens = True

        self.context_window_length = self.tokenizer.model_max_length
        if self.context_window_length > 1000000:
            if hasattr(self.model.config, 'n_positions'):
                self.context_window_length = self.model.config.n_positions
            elif hasattr(self.model.config, 'max_position_embeddings'):
                self.context_window_length = self.model.config.max_position_embeddings
            else:
                print('Model has unclear context_window_length, setting to 1024')
                self.context_window_length = 1024 


    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200, max_length: int = 1024
    ) -> List[str]:
        if self.temperature == 0:
            assert not do_sample
            assert num_samples == 1

        prompt = prompt.replace("    ", "\t")
        input_tokens = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        scores = StoppingCriteriaList(
            [
                EndOfFunctionCriteria(
                    start_length=len(input_tokens[0]),
                    eos=self.eos,
                    tokenizer=self.tokenizer,
                )
            ]
        )


        max_new_tokens = min(max_length, self.context_window_length - len(input_tokens[0]))     

        while max_new_tokens > 0:
            try:
                raw_outputs = self.model.generate(
                    **input_tokens,
                    decoder_input_ids=input_tokens["input_ids"],
                    max_new_tokens=max_new_tokens,
                    stopping_criteria=scores,
                    do_sample=do_sample,
                    top_p=0.95,
                    top_k=None,
                    temperature=self.temperature,
                    output_scores=True,
                    return_dict_in_generate=True,
                    num_return_sequences=min(self.batch_size, num_samples),
                    pad_token_id=self.tokenizer.eos_token_id,
                    decoder_start_token_id=self.tokenizer.pad_token_id,
                )  # remove warning
            except RuntimeError as e:  # catch torch OOM
                if "CUDA out of memory" in str(e):
                    old_max_new_tokens = max_new_tokens
                    max_new_tokens = int(max_new_tokens * 0.8)
                    print(
                        f"OOM, reducing max_new_tokens from {old_max_new_tokens} to {max_new_tokens}"
                    )
                    continue
                else:
                    raise e

            break
        gen_seqs = raw_outputs.sequences[:, len(input_tokens[0]) :]
        gen_strs = self.tokenizer.batch_decode(
            gen_seqs, skip_special_tokens=self.skip_special_tokens
        )
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    # could be multiple eos in outputs, better pick minimum one
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index].replace("\t", "    "))
        return outputs


def make_model(name: str, batch_size: int = 1, temperature: float = 0.8, ngpus: int = 1):
    if name == "codegen-2b":
        return HFTorchDecoder(
            batch_size=batch_size,
            name="Salesforce/codegen-2B-mono",
            temperature=temperature,
        )
    elif name == "codegen-6b":
        return HFTorchDecoder(
            batch_size=batch_size,
            name="Salesforce/codegen-6B-mono",
            temperature=temperature,
        )
    elif name == "codegen-16b":
        return HFTorchDecoder(
            batch_size=batch_size,
            name="Salesforce/codegen-16B-mono",
            temperature=temperature,
        )
    elif name == "codegen2-1b":
        return Codegen2Decoder(
            batch_size=batch_size,
            name="Salesforce/codegen2-1B",
            temperature=temperature,
        )
    elif name == "codegen2-3b":
        return Codegen2Decoder(
            batch_size=batch_size,
            name="Salesforce/codegen2-3_7B",
            temperature=temperature,
        )
    elif name == "codegen2-7b":
        return Codegen2Decoder(
            batch_size=batch_size,
            name="Salesforce/codegen2-7B",
            temperature=temperature,
        )
    elif name == "codegen2-16b":
        warn(
            "codegen2-16b checkpoint is `unfinished` at this point (05/11/2023) according to their paper. "
            "So it might not make sense to use it."
        )
        return Codegen2Decoder(
            batch_size=batch_size,
            name="Salesforce/codegen2-16B",
            temperature=temperature,
        )
    elif name == "polycoder":
        return HFTorchDecoder(
            batch_size=batch_size,
            name="NinedayWang/PolyCoder-2.7B",
            temperature=temperature,
        )
    elif name == "santacoder":
        return SantaCoder(
            batch_size=batch_size, name="bigcode/santacoder", temperature=temperature
        )
    elif name == "incoder-1b":
        return IncoderDecoder(
            batch_size=batch_size, name="facebook/incoder-1B", temperature=temperature
        )
    elif name == "incoder-6b":
        return IncoderDecoder(
            batch_size=batch_size, name="facebook/incoder-6B", temperature=temperature
        )
    elif name == "stablelm-7b":
        return HFTorchDecoder(
            batch_size=batch_size,
            name="StabilityAI/stablelm-base-alpha-7b",
            temperature=temperature,
        )
    elif name.startswith("gpt-3.5-") or name.startswith("gpt-4"):
        return OpenAIChatDecoder(
            batch_size=batch_size,
            name=name,
            temperature=temperature,
            conversational=True,
        )
    elif name.startswith("claude"):
        return AnthropicMessageDecoder(
            batch_size=batch_size,
            name=name,
            temperature=temperature,
        )
    elif name.startswith("gemini"):
        return GeminiDecoder(
            batch_size=batch_size,
            name=name,
            temperature=temperature,
        )    
    elif name == "gptneo-2b":
        return HFTorchDecoder(
            batch_size=batch_size,
            name="EleutherAI/gpt-neo-2.7B",
            temperature=temperature,
        )
    elif name == "gpt-j":
        return HFTorchDecoder(
            batch_size=batch_size, name="EleutherAI/gpt-j-6B", temperature=temperature
        )
    elif name.startswith("starcoder"):
        # return VLlmDecoder( #why ?
        #     batch_size=batch_size,
        #     name=f"bigcode/{name}",
        #     temperature=temperature,
        #     tensor_parallel_size=ngpus,
        # )
        return StarCoderInfill( 
            # batch_size=batch_size, name=f"bigcode/{name}", temperature=temperature
            batch_size=batch_size, name=name, temperature=temperature
        )
    elif name.startswith("granite"):
        return HFTorchDecoder(
            batch_size=batch_size, name=name, temperature=temperature
        )
    elif name.startswith("codellama"):#CodeLlama-13b-Instruct-hf
        assert name.endswith("hf")
        nb = name.split("-")[-3]
        return HFTorchDecoder(
            batch_size=batch_size, name=f"CodeLlama-{nb}-Instruct-hf", temperature=temperature
        )
    elif name.startswith("code-llama-"):
        assert name.endswith("b")
        nb = name.split("-")[-1]
        return VLlmDecoder(
            batch_size=batch_size,
            name=f"codellama/CodeLlama-{nb}-Instruct-hf",
            temperature=temperature,
        )
    elif name == "codet5p-2b":
        return CodeT5P(
            batch_size=batch_size,
            name="Salesforce/codet5p-2b",
            temperature=temperature,
        )
    elif name == "codet5p-6b":
        return CodeT5P(
            batch_size=batch_size,
            name="Salesforce/codet5p-6b",
            temperature=temperature,
        )
    elif name == "codet5p-16b":
        return CodeT5P(
            batch_size=batch_size,
            name="Salesforce/codet5p-16b",
            temperature=temperature,
        )
    elif name.startswith("deepseek-coder"):
        import re

        # format deepseek-coder-{nb}b*
        pattern = re.compile(r"deepseek-coder-(\d+\.?\d*)b(.*)")
        matches = pattern.findall(name)[0]
        nb = float(matches[0])
        if nb.is_integer():
            nb = int(nb)

        if "instruct" in name:
            return DeepSeekInstruct(
                batch_size=batch_size,
                name=f"deepseek-ai/deepseek-coder-{nb}b-instruct",
                temperature=temperature,
                conversational=True,
            )
        else:
            return VLlmDecoder(
                batch_size=batch_size,
                name=f"deepseek-ai/deepseek-coder-{nb}b-base",
                temperature=temperature,
                tensor_parallel_size=ngpus,
            )
    elif name == "wizardcoder-33b":
        return VLlmDecoder(
            batch_size=batch_size,
            name="WizardLM/WizardCoder-33B-V1.1",
            temperature=temperature,
            conversational=True,
            tensor_parallel_size=ngpus,
        )    
    elif name == "wizardcoder-34b":
        return VLlmDecoder(
            batch_size=batch_size,
            name="WizardLM/WizardCoder-Python-34B-V1.0",
            temperature=temperature,
            conversational=True,
            tensor_parallel_size=ngpus,
        )
    elif name == "wizardcoder-15b":
        return VLlmDecoder(
            batch_size=batch_size,
            name="WizardLM/WizardCoder-15B-V1.0",
            temperature=temperature,
            conversational=True,
            tensor_parallel_size=ngpus,
        )
    elif name == "wizardcoder-7b":
        return VLlmDecoder(
            batch_size=batch_size,
            name="WizardLM/WizardCoder-Python-7B-V1.0",
            temperature=temperature,
            conversational=True,
            tensor_parallel_size=ngpus,
        )
    elif name == "mistral-7b-codealpaca":
        return HFTorchDecoder(
            batch_size=batch_size,
            name="Nondzu/Mistral-7B-codealpaca-lora",
            temperature=temperature,
        )
    elif name == "zephyr-7b":
        return HFTorchDecoder(
            batch_size=batch_size,
            name="HuggingFaceH4/zephyr-7b-beta",
            temperature=temperature,
        )
    elif name == "codebooga-34b":
        return HFTorchDecoder(
            batch_size=batch_size,
            name="oobabooga/CodeBooga-34B-v0.1",
            temperature=temperature,
        )
    elif name == "phind-code-llama-34b-v2":
        return VLlmDecoder(    
            batch_size=batch_size,
            name="Phind/Phind-CodeLlama-34B-v2",
            temperature=temperature,
            tensor_parallel_size=ngpus,
        )
    elif name == "mistral-7b":
        return HFTorchDecoder(
            batch_size=batch_size,
            name="mistralai/Mistral-7B-v0.1",
            temperature=temperature,
        )
    elif name == "dolphin-2.6":
        return ChatML(
            batch_size=batch_size,
            name="cognitivecomputations/dolphin-2.6-mixtral-8x7b",
            temperature=temperature,
            tensor_parallel_size=ngpus,
            # max_new_tokens=512 + 256,
        )
    elif name == "solar-10.7b-instruct":
        return ChatML(
            batch_size=batch_size,
            name="upstage/SOLAR-10.7B-Instruct-v1.0",
            temperature=temperature,
            conversational=True,
        )
    elif name == "mistral-hermes-codepro-7b":
        return ChatML(
            batch_size=batch_size,
            name="beowolx/MistralHermes-CodePro-7B-v1",
            temperature=temperature,
            # max_new_tokens=512 + 256,
        )
    elif name == "phi-2":
        return VLlmDecoder(    
            batch_size=batch_size,
            name="microsoft/phi-2",
            temperature=temperature,
            tensor_parallel_size=ngpus,
        )
    elif name == "mixtral-8x7b-instruct":
        return VLlmDecoder(
            batch_size=batch_size,
            name="mistralai/Mixtral-8x7B-Instruct-v0.1",
            temperature=temperature,
            tensor_parallel_size=ngpus,
        )
    elif name == "octocoder":
        return VLlmDecoder(
            batch_size=batch_size,
            name="bigcode/octocoder",
            temperature=temperature,
            tensor_parallel_size=ngpus,
        )
    elif name == "magicoder-s-ds-6.7b":
        return VLlmDecoder(
            batch_size=batch_size,
            name="ise-uiuc/Magicoder-S-DS-6.7B",
            temperature=temperature,
            tensor_parallel_size=ngpus,
        )
    elif name == "magicoder-s-cl-7b":
        return VLlmDecoder(
            batch_size=batch_size,
            name="ise-uiuc/Magicoder-S-CL-7B",
            temperature=temperature, 
            tensor_parallel_size=ngpus, 
        )

    raise ValueError(f"Invalid model name: {name}")
