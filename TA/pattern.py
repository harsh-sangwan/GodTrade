import csv


def find_hammer(li):
    tol = 1
    o = float(li[1])
    h = float(li[2])
    l = float(li[3])
    c = float(li[4])
    if(o<c):      #bullish
        diff = h-l
        body = c - o
        head = h - c
        tail = o - l
        if(head <=tol*diff/100 and tail>2*body):
            print(li)    
    else:          #bearish
        diff = h-l
        body = o - c
        head = h - o
        tail = c - l
        if(head <=tol*diff/100 and tail>2*body):
            print(li)


def find_invhammer(li):
    tol = 1
    o = float(li[1])
    h = float(li[2])
    l = float(li[3])
    c = float(li[4])
    if(o<c):   #bullish
        diff = h-l
        body = c - o
        head = h - c
        tail = o - l
        if(tail <=tol*diff/100 and head>2*body):
            print(li)    
    else:     #bearish
        diff = h-l
        body = o - c
        head = h - o
        tail = c - l
        if(tail <=tol*diff/100 and head>2*body):
            print(li)

def find_bulleng(li1,li2):

    o1 = float(li1[1])
    h1 = float(li1[2])
    l1 = float(li1[3])
    c1 = float(li1[4])
    o2 = float(li2[1])
    h2 = float(li2[2])
    l2 = float(li2[3])
    c2 = float(li2[4])
    if(o1>c1 and o2<c2):
        if(o2<c1 and c2>o1):
            print(li1)
            print(li2)
            print("\n")    
    
def find_beareng(li1,li2):
    
    o1 = float(li1[1])
    h1 = float(li1[2])
    l1 = float(li1[3])
    c1 = float(li1[4])
    o2 = float(li2[1])
    h2 = float(li2[2])
    l2 = float(li2[3])
    c2 = float(li2[4])
    if(o1<c1 and o2>c2):
        if(o2>c1 and c2<o1):
            print(li1)
            print(li2)
            print("\n")

def piercing(li1,li2):
#check for downtrend compulsory in the main program minimum of 5 entries  
    tol = 2
    o1 = float(li1[1])
    h1 = float(li1[2])
    l1 = float(li1[3])
    c1 = float(li1[4])
    o2 = float(li2[1])
    h2 = float(li2[2])
    l2 = float(li2[3])
    c2 = float(li2[4])
    if(o1>c1 and o2<c2 and o1-c1> max(4,tol*o1/100) ):
        if(c2 > l1+((h1-l1)/2) and o2<c1):
            print(li1)
            print(li2)
            print("\n")
            
def bearharami(li1,li2):
#check for uptrend compulsory in the main program minimum of 5 entries  
    tol = 2
    o1 = float(li1[1])
    h1 = float(li1[2])
    l1 = float(li1[3])
    c1 = float(li1[4])
    o2 = float(li2[1])
    h2 = float(li2[2])
    l2 = float(li2[3])
    c2 = float(li2[4])
    if(o1<c1 and o2>c2 and c1-o1> max(5,tol*o1/100) ):
        if(c1>o2 and c2>o1):
            print(li1)
            print(li2)
            print("\n")

def bullharami(li1,li2):
#check for downtrend compulsory in the main program minimum of 5 entries  
    tol = 2
    o1 = float(li1[1])
    h1 = float(li1[2])
    l1 = float(li1[3])
    c1 = float(li1[4])
    o2 = float(li2[1])
    h2 = float(li2[2])
    l2 = float(li2[3])
    c2 = float(li2[4])
    if(o1>c1 and o2<c2 and o1-c1> max(5,tol*o1/100) ):
        if(c1<o2 and c2<o1):
            print(li1)
            print(li2)
            print("\n")

def evenstar(li1,li2,li3):
#Check for uptrend compulsory    
    tol = 1
    o1 = float(li1[1])
    h1 = float(li1[2])
    l1 = float(li1[3])
    c1 = float(li1[4])
    o2 = float(li2[1])
    h2 = float(li2[2])
    l2 = float(li2[3])
    c2 = float(li2[4])
    o3 = float(li3[1])
    h3 = float(li3[2])
    l3 = float(li3[3])
    c3 = float(li3[4])
    if(c1>o1 and c2>c1 and o2>c1 and abs(o2-c2)< min(1,tol*o1/100) and o3>c3):
         if(c3 < (o1+c1)/2 and h2!=l2):
            print(li1)
            print(li2)
            print(li3)
            print("\n")
    
def mornstar(li1,li2,il3):
#Check for downtrend
    tol = 1
    o1 = float(li1[1])
    h1 = float(li1[2])
    l1 = float(li1[3])
    c1 = float(li1[4])
    o2 = float(li2[1])
    h2 = float(li2[2])
    l2 = float(li2[3])
    c2 = float(li2[4])
    o3 = float(li3[1])
    h3 = float(li3[2])
    l3 = float(li3[3])
    c3 = float(li3[4])
    if(c1<o1 and c2<c1 and o2<c1 and c3 > (o1+c1)/2 and o3<c3):
         if(abs(o2-c2)< min(1,tol*o1/100) and h2!=l2):
            print(li1)
            print(li2)
            print(li3)
            print("\n")
               

def doji(li1,li2):
    tol = 2
    o1 = float(li1[1])
    h1 = float(li1[2])
    l1 = float(li1[3])
    c1 = float(li1[4])
    o2 = float(li2[1])
    h2 = float(li2[2])
    l2 = float(li2[3])
    c2 = float(li2[4])    
    if(o1>c1 and h1!=l1):  #bearish
        if(o1-c1 > max(5,tol*o1/100) and o2<c1 and c2<c1 and abs(o2-c2)<0.25):
            print(li1)
            print(li2)
            print("Morning\n")
    elif(h1!=l1):
        if(c1-o1 > max(5,tol*o1/100) and o2>c1 and c2>c1 and abs(o2-c2)<0.25):
            print(li1)
            print(li2)
            print("Evening\n")
           

if __name__ == "__main__":

    with open('table.csv', 'r') as f:
        reader = csv.reader(f)
        dat = list(reader)

    for i in range(2,len(dat)-1):
        #li3 = dat[i-2]
        li2 = dat[i-1]
        li1 = dat[i]
        doji(li1,li2)       
        
