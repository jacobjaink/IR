#Python 2.7.3
import re
import os
import collections
import time

class index:
	def __init__(self,path):
		pass
		self.loc=path
		self.File=open("output.txt", 'w')
		
	'''
	First the tokens are generated one by one for each file and simultaneously a dictionary pos_idx is stored
	with all token values and their corresponding positions as a list of dictionaries inside this dictionary.
	When similar words are encountered in other files , the existing dictionary value for the word is appended.
	In case of new words coming in the file, new keys are added to the dictionary. The docId is also generated for each
	file during this process by simply numbering from 1 to n.
	'''			
	def buildIndex(self):
		#function to read documents from collection, tokenize and build the index with tokens
		#index should also contain positional information of the terms in the document --- term: [(ID1,[pos1,pos2,..]), (ID2, [pos1,pos2,…]),….]
		#use unique document IDs
		pass
		#print(self.loc)
		current = time.time()
		docList=[]
		pos_idx={}
		dic={}
		docId=0
		vocab={}
		for filename in os.listdir(self.loc):
			docId+=1
			docList.append((docId,filename))			
			fullfilename = os.path.join(self.loc, filename)
			file = open(fullfilename, 'r')
			s=file.read().lower()
			tokens=re.findall('[a-z]+', s)
			
			'''
			if(docId==153):
				print(s)
				y=[]
				for i,t in enumerate(tokens):
					y.append((t,i))
				print(y)
			'''
			for i,t in enumerate(tokens):
				dic={}
				if t not in vocab:
					vocab[t]=[docId]
					dic[docId]=[i+1]
					pos_idx[t]=[dic]
				elif docId not in vocab[t]:
					vocab[t].append(docId)
					dic[docId]=[i+1]
					pos_idx[t].append(dic)
				else:
					for x in pos_idx[t]:
						if docId in x:
							x[docId].append(i+1)			
		
		self.posting=pos_idx
		self.docIdx=docList
		end = time.time()
		diff = end - current
		#print(pos_idx)
		self.File.write("Index built in "+str(diff)+" seconds.\n")
		print("Index built in "+str(diff)+" seconds.")
		
		
	'''
	The list of documents for each query terms provided is fetched and analyzed in sorted order.
	If the values of two list match, we got a hit on the intersecting document which lets both the list pointers to move
	forward. In all unequal case, the pointer with smaller doc value moves forward. This makes sure the complexity of O(m+n)
	'''
	def and_query(self, query_terms):
		#function for identifying relevant docs using the index
		pass
		current = time.time()
		i=1		
		l=sorted([int(k) for x in self.posting.get(query_terms[0].lower(),[]) for k,v in x.items()])
		if (len(query_terms)>=2):
			while i < len(query_terms):
				temp=sorted([int(k) for x in self.posting.get(query_terms[i].lower(),[]) for k,v in x.items()])
				l1=len(l)
				l2=len(temp)
				pt1=0
				pt2=0
				lis=[]
				while(pt1<l1) and (pt2<l2):
					if(l[pt1]==temp[pt2]):
						lis.append(l[pt1])
						pt2=pt2+1
						pt1=pt1+1
					elif l[pt1]>temp[pt2]:
						pt2=pt2+1
					else:
						pt1=pt1+1
				i+=1
				l=lis
		end = time.time()
		diff = end - current
		self.File.write("\nResults for the Query: "+" AND ".join(query_terms)+"\nTotal Docs retrieved: "+str(len(l)))		
		print("Results for the Query: "+" AND ".join(query_terms)+"\nTotal Docs retrieved: "+str(len(l)))
		for j in l:
			self.File.write("\n"+str(self.docIdx[j-1][1]))
			print(self.docIdx[j-1][1])
		self.File.write("\nRetrieved in "+str(diff)+" seconds\n")	
		print("Retrieved in "+str(diff)+" seconds")		
	
	'''
	This prints out the dictionary value generate during indexing. The tokens are the keys and values contain a list of
	docId and postings (word location as a list) as a tuple across the entire documents present.
	'''
	def print_dict(self):
		#function to print the terms and posting list in the index
		pass
		self.File.write("\n")
		for k,v in (sorted(self.posting.items(), key=lambda x: x[0])):
			self.File.write("\n")
			self.File.write(k)
			print(k, end=" ")
			for i in v:
				for x,y in i.items():
					self.File.write(str((x,y)))
					print((x,y), end=" ")
			print()
		
	'''
	This is list containing the tuples with unique docId and the corresponding file names
	'''
	def print_doc_list(self):
		# function to print the documents and their document id
		pass
		self.File.write("\n")		
		for x in self.docIdx:
			self.File.write("\nDoc ID: "+str(x[0])+" ==> "+str(x[1]))
			print("Doc ID: "+str(x[0])+" ==> "+str(x[1]))
			
'''
Used for running the functions inside the class by creating an object of the class.
'''	
		
def main():
	a=index('./collection')
	a.buildIndex()
	'''
	a.and_query(['with','without','yemen'])
	a.and_query(['with','without','yemen','yemeni'])
	a.and_query(['meeting','agricultural','experts'])
	a.and_query(['antigovernment','meetings'])
	a.and_query(['Ransacked','government','offices','burned','cars'])
	a.print_dict()
	a.print_doc_list()	
	'''	
if __name__ == '__main__':
	main()
    	