#Python 3.0
import re
import os
import collections
import time
import numpy as np
#import other modules as needed

class pagerank:
	
		
	def pagerank(self, input_file, alpha=0.15, iteration=20, tol=.001):
		#function to implement pagerank algorithm
		#input_file - input file that follows the format provided in the assignment description
		pass
		nop=0
		nol=0
		srcdest=[]
		i=0	
		with open(input_file,'r') as f:
			content= [x.strip('\n') for x in f.readlines()]
		
		for x in content:
			if(i==0):
				nop=int(x)
			elif i==1:
				nol=int(x)
			else:
				data=x.split()
				srcdest.append((int(data[0]),int(data[1])))
			i+=1
		#print(nop,nol,srcdest)
		'''
		Creating Transition Matrix
		'''
		P=[]
		for x in range(nop):
			P.append([])
			for y in range(nop):
				P[x].append(0)
				
		for (x, y) in srcdest:
			P[x][y]=1
		#print(P)
		
		a=alpha
		for x in range(nop):
			s=sum(P[x])
			for y in range(nop):
				if s==0:
					P[x][y]=1/nop
				else:
					P[x][y]= (((P[x][y]/s)*(1-a)))+(a/nop)
		#print(P)
		
		X=[]
		for x in range(nop):
			X.append(1/nop)
			
		iter=iteration
		
		for x in range(iter):
			X_old=X
			X=np.dot(X,P)
			X_old=X-X_old
			tolerance=True
			for i in X_old:
				if abs(i)>tol:
					tolerance=False
			if tolerance:
				iter=x
				break
		#print(X)
		print("PageId and Page Rank for file:",input_file)
		print("alpha="+str(a)+" iterations="+str(iter)+" Tolerance for rank value to reduce the default 20 iteration:"+str(tol))
		print(sorted(list(zip(range(nop),X)), key=lambda x:-x[1])[:5])
		
		File=open("out.txt","a")
		File.write("PageId and Page Rank for file:"+input_file)
		File.write("\nalpha="+str(a)+" iterations="+str(iter)+" Tolerance for rank value to reduce the default 20 iteration:"+str(tol))
		File.write("\n"+str(sorted(list(zip(range(nop),X)), key=lambda x:-x[1])[:5])+"\n\n")
def main():
	a=pagerank()
	a.pagerank('test1.txt')
	a.pagerank('test2.txt')
	a.pagerank('test3.txt',alpha=0.14,iteration=13)
if __name__ == '__main__':
	main()