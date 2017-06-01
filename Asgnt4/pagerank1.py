# Python 3.0
from collections import defaultdict
# from sortedcontainers import SortedSet
import re
import os
import collections
import time


class pagerank:
	def __init__(self, path):
		self.path = '.'

	def read_doc(self):
		with open('test3.txt') as f:
			input_file = f.read()
		
		n = input_file[0]

		inlinks = defaultdict(list)
		outlinks = defaultdict(list)

		for line in open('test3.txt').readlines():
			lis = list(map(int, line.split()))
			if len(lis) > 1:
				u = lis[0]
				outlinks[u].append(lis[1])
		# print ("outlinks", outlinks)

		for line in open('test3.txt').readlines():
			lis = list(map(int, line.split()))
			if len(lis) > 1:
				u = lis[1]
				inlinks[u].append(lis[0])
		# print ("inlinks:", inlinks)
		return input_file, inlinks, outlinks

	def page_rank(self, input_file, inlinks, outlinks):

		pages = []
		for value in outlinks:
			pages.append(value)

		for value in inlinks:
			pages.append(value)

		pages_list = list(set(pages))
		# print (pages_list)

		b = 0.14
		iters = 13

		P_R = defaultdict(lambda: 1.0)
		for p in pages_list:
			P_R[p] = 1.0 / len(pages_list)
		P_R_new=defaultdict()	
			
		for i in range(iters):
			for p in pages_list:
				sum_ = 0.0
				for w in pages_list:
					if w in inlinks[p]:
						if len(outlinks[w])!=0:
							const1= 1.0 / len(outlinks[w])
						else:
							const1= 1.0 / len(pages_list)
					else:
						const1=0
					const1 =(b / len(pages_list)) + ((1.0 - b) * const1)
					
					sum_+=const1*P_R[w]
				P_R_new[p]=sum_
			P_R=P_R_new	
		# print (P_R)
		sort_dic = sorted(P_R.items(), key=lambda x:x[1], reverse=True)
		print (sort_dic)
		# lis_k = []
		# lis_v = []
		# for k,v in sort_dic:
		# 	lis_k.append(k)
		# 	lis_v.append(v)

		# output= open('out.txt', 'w')
		# output.write(lis_k)
		# return sort_dic

def main():
	path = "/Users/Vidhy/Desktop/COURSES/IR/IR_ASG4"
	pr = pagerank(path)
	input_file, inlinks, outlinks = pr.read_doc()
	rank = pr.page_rank(input_file, inlinks, outlinks)
if __name__ == '__main__':
    main()

	
