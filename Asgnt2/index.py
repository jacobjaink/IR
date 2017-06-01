#Python 3.0
import re
import os
import collections
import time
import math
#import other modules as needed
import random
import matplotlib.pyplot as plt
import contextlib
import sys
from timeit import default_timer as timer

@contextlib.contextmanager
def nostdout():
	save_stdout = sys.stdout
	sys.stdout = open('trash','w')
	yield
	sys.stdout = save_stdout
	
class index:
	def __init__(self,path):
		self.loc=path
		self.File=open("output.txt", 'w')
	def buildIndex(self):
		#function to read documents from collection, tokenize and build the index with tokens
		# implement additional functionality to support methods 1 - 4
		#use unique document integer IDs
		current = time.time()
		docList=[]
		pos_idx={}
		dic={}
		docId=0
		vocab={}
		stopfile=open('.\stop-list.txt','r')
		stoplist=stopfile.read().lower()
		stoptokens=re.split('\s',stoplist)
		#print(stoptokens)
		
		'''
		creating index with term frequency and default IDF as zero
		'''
		for filename in os.listdir(self.loc):
			docId+=1
			docList.append((docId,filename))			
			fullfilename = os.path.join(self.loc, filename)
			file = open(fullfilename, 'r')
			s=file.read().lower()
			tokens=list(re.findall('[a-z]+', s))
			#print(tokens)
			tokens=[x for x in tokens if x not in stoptokens]
			
			for i,t in enumerate(tokens):
				dic={}
				if t not in vocab:
					vocab[t]=[docId]
					dic[docId]=[1]
					dic[docId].append(i+1)
					pos_idx[t]=[{'IDF':0}]
					pos_idx[t].append(dic)
				elif docId not in vocab[t]:
					vocab[t].append(docId)
					dic[docId]=[1]
					dic[docId].append(i+1)
					pos_idx[t].append(dic)
				else:					
					for x in pos_idx[t]:
						if docId in x:
							x[docId][0]+=1
							x[docId].append(i+1)			
		#print(vocab)
		#print(docId)
		#print(sorted(pos_idx.items(), key=lambda x: x[0]))
		self.vocab=vocab.keys()
		'''
		setting the IDF value and the weighted term frequency
		'''
		
		for k,v in pos_idx.items():
			v[0]['IDF']=math.log10(docId/(len(v)-1))
			for i in v:
				for x,y in i.items():
					if x!='IDF':
						y[0]=1+(math.log10(y[0]))
						#print(y[0])
		
		'''
		Finding Doc Length of each document
		'''
		doc_length={}
		avg_len=0
		for k,v in pos_idx.items():
			v[0]['IDF']=math.log10(docId/(len(v)-1))
			avg_len+=len(v)-1
			for i in v:
				for x,y in i.items():
					if x!='IDF':
						doc_length[x]=doc_length.get(x,0)+(y[0]*v[0]['IDF'])**2
		
		#print(doc_length)
		avg_len/=len(pos_idx)
		#print(round(avg_len))
		doc_length={k:math.sqrt(v) for k,v in doc_length.items()}
		#print(set([x for x in range(1,423)])-set(doc_length.keys()))			
		self.posting=pos_idx
		self.docIdx=docList
		self.doclen=doc_length
		end = time.time()
		diff = end - current
		print("Index built in "+str(diff)+" seconds.")
		self.File.write("Index built in "+str(diff)+" seconds.\n")
		
		current=time.time()
		'''
		create Champion's list
		'''
		champ_list={}
		'''
		champlist is created by selecting only the documents with highest term frequency
		'''
		mean_Idf=0
		for key,v in pos_idx.items():
			champ_list[key]=[(v[0]['IDF'],float('inf'))]
			mean_Idf+=v[0]['IDF']
			for x in v:
				for k1,v1 in x.items():
					if k1!='IDF':
						champ_list[key].append((k1,v1[0]))
		
		for key,v in champ_list.items():
			v.sort(key=lambda x:-x[1])
		
		
		mean_Idf/=len(pos_idx)
		'''
		filtering out top 10 records with highest tfidf or the rarest terms with all posting list
		'''
		
		for key,v in champ_list.items():
			v=[x for c,x in enumerate(v) if c<max(11,round(avg_len)+1) or v[0][0]>1.1*mean_Idf]
			champ_list[key]=v
		
		self.champ_list=champ_list
		end = time.time()
		diff = end - current
		self.File.write("Champion list built in "+str(diff)+" seconds.\n")
		print("Champion list built in "+str(diff)+" seconds.")
		
		
		current=time.time()
		'''
		Creating sqrt(N) random leaders from N documents
		'''
		l=[x for x in range(1,len(self.docIdx)+1)]
		leadlist=random.sample(l,int(math.sqrt(len(self.docIdx)+1)))
		leadlist=[x[0] for c,x in enumerate(self.docIdx) if c+1 in leadlist]
		doc_idf={}
		
		for key,v in self.posting.items():
			for x in v:
				for k1,v1 in x.items():
					if k1!='IDF':
						if k1 not in doc_idf:
							tfdf=v1[0]*v[0]['IDF']
							doc_idf[k1]={key:tfdf}
						else:
							tfdf=v1[0]*v[0]['IDF']
							doc_idf[k1][key]=tfdf
		
		leaders={k:v for k,v in doc_idf.items() if k in leadlist}
		followers={k:v for k,v in doc_idf.items() if k not in leadlist}
	
		'''
		calculating the cosine similarity for documents selected as leaders vs the followers to cluster them
		'''
		cluster={}
		doc=0
		for d1,tfdf1 in followers.items():
			sim_min=float('inf')
			for d2,tfdf2 in leaders.items():
				cos_sim=0
				for word in (set(tfdf1.keys()) & set(tfdf2.keys())):
					cos_sim+=tfdf1[word]*tfdf2[word]
				cos_sim=cos_sim/(self.doclen[d1]*self.doclen[d2])
				if(cos_sim<sim_min):
					sim_min=cos_sim
					doc=d2
			if doc not in cluster:
				cluster[doc]=[d1]
			else:
				cluster[doc].append(d1)
		
		end = time.time()
		diff = end - current
		self.File.write("Random leaders and clusters built in "+str(diff)+" seconds.")
		print("Random leaders and clusters built in "+str(diff)+" seconds.")
		self.doc_idf=doc_idf
		self.leaders=leaders
		self.cluster=cluster
	def exact_query(self, query_terms, k):
	#function for exact top K retrieval (method 1)
	#Returns at the minimum the document names of the top K documents ordered in decreasing order of similarity score
		pass
		current = timer()
	
		'''
		compute vector form of query
		'''
		q=collections.Counter(query_terms)
		for x,y in q.items():
			#print(self.posting.get(x,[{'IDF':0}])[0]['IDF'])
			q[x]=(1+(math.log10(q[x])))*self.posting.get(x,[{'IDF':0}])[0]['IDF']
		qdist=math.sqrt(sum((q[k])**2 for k in q.keys()))
		'''
		normalizing the query vector
		'''
		if(qdist!=0):
			q={k:v/qdist for k,v in q.items()}
		else:
			q={k:0 for k,v in q.items()}
		
		'''
		calculating the document score with respect to each query
		'''
		doc_score={}
		for term,qwt in q.items():
			l=self.posting.get(term,[])
			for p in l:
				for d,dwt in p.items():
					if d!='IDF':
						doc_score[d]=doc_score.get(d,0)+(dwt[0]*l[0]['IDF']*qwt) 
		
		'''
		normalizing the document score with document length
		'''
		doc_score={k:v/self.doclen[k] for k,v in doc_score.items()}
		doc_score=sorted(doc_score.items(), key=lambda x: -x[1])
		end = timer()
		diff = end - current
		self.exact_time=diff
		self.File.write("\n\nExact query search in "+str(diff)+" seconds.\n")
		print("Exact query search in "+str(diff)+" seconds.")
		'''
		printing the top k documents based on cosine similarity
		'''
		self.File.write("Result for exact query search for query:"+str(query_terms))
		self.File.write("\n")
		print("Result for exact query search for query:",query_terms)
		exact=[]
		for count,x in enumerate(doc_score):
			if(count<k):
				self.File.write(self.docIdx[x[0]-1][1])
				self.File.write("\n")
				exact.append(self.docIdx[x[0]-1][1])
				print(self.docIdx[x[0]-1][1])
		self.exact=exact
	def inexact_query_champion(self, query_terms, k):
	#function for exact top K retrieval using champion list (method 2)
	#Returns at the minimum the document names of the top K documents ordered in decreasing order of similarity score
		pass
		current=timer()
		'''
		Transforming query to normalized vector
		'''	
		q=collections.Counter(query_terms)
		for x,y in q.items():
			q[x]=(1+(math.log10(q[x])))*self.posting.get(x,[{'IDF':0}])[0]['IDF']
		qdist=math.sqrt(sum((q[key])**2 for key in q.keys()))
		if(qdist!=0):
			q={key:v/qdist for key,v in q.items()}
		else:
			q={key:0 for key,v in q.items()}
		
		'''
		Top champ list is considered for document score generation
		'''
		doc_score={}
		for term,qwt in q.items():
			l=self.champ_list.get(term,[])
			for c,(d,dwt) in enumerate(l):
				if c!=0:
					doc_score[d]=doc_score.get(d,0)+(dwt*l[0][0]*qwt) 
				
					
		doc_score={key:v/self.doclen[key] for key,v in doc_score.items()}
		doc_score=sorted(doc_score.items(), key=lambda x: -x[1])
		#print(doc_score)
		end = timer()
		diff = end - current
		self.champ_time=diff
		self.File.write("\n\nInexact champion query search in "+str(diff)+" seconds.\n")
		print("Inexact champion query search in "+str(diff)+" seconds.")
		self.File.write("Result for inexact champion search for query:"+str(query_terms))
		self.File.write("\n")
		print("Result for inexact champion search for query:",query_terms)
		inexact_champion=[]
		for count,x in enumerate(doc_score):
			if(count<k):
				self.File.write(self.docIdx[x[0]-1][1])
				self.File.write("\n")
				print(self.docIdx[x[0]-1][1])
				inexact_champion.append(self.docIdx[x[0]-1][1])
		self.inexact_champion=inexact_champion	
	def inexact_query_index_elimination(self, query_terms, k):
	#function for exact top K retrieval using index elimination (method 3)
	#Returns at the minimum the document names of the top K documents ordered in decreasing order of similarity score
		pass
		current=timer()
		'''
		Creating normalized query vector
		'''
		q=collections.Counter(query_terms)
		for x,y in q.items():
			#print(self.posting.get(x,[{'IDF':0}])[0]['IDF'])
			q[x]=(1+(math.log10(q[x])))*self.posting.get(x,[{'IDF':0}])[0]['IDF']
		qdist=math.sqrt(sum((q[k])**2 for k in q.keys()))
		if(qdist!=0):
			q={k:v/qdist for k,v in q.items()}
		else:
			q={k:0 for k,v in q.items()}
		
		q=sorted(q.items(),key=lambda k:-k[1])
		'''
		filtering out half of the query terms which have low tfIDF value
		'''
		q={k:v for c,(k,v) in enumerate(q) if c<len(q)/2}
		
		'''
		Finding the document score with half the query terms for each document
		'''
		doc_score={}
		for term,qwt in q.items():
			l=self.posting.get(term,[])
			for p in l:
				for d,dwt in p.items():
					if d!='IDF':
						doc_score[d]=doc_score.get(d,0)+(dwt[0]*l[0]['IDF']*qwt) 
		
		doc_score={k:v/self.doclen[k] for k,v in doc_score.items()}
		doc_score=sorted(doc_score.items(), key=lambda x: -x[1])
		#print(doc_score)
		end = timer()
		diff = end - current
		self.idx_time=diff
		self.File.write("\n\nInexact index elimination search in "+str(diff)+" seconds.\n")
		print("Inexact index elimination search in "+str(diff)+" seconds.")
		self.File.write("Result for inexact index elimination search for query:"+str(query_terms))
		self.File.write("\n")
		print("Result for inexact index elimination search for query:",query_terms)
		inexact_idx_elim=[]
		for count,x in enumerate(doc_score):
			if(count<k):
				self.File.write(self.docIdx[x[0]-1][1])
				self.File.write("\n")
				print(self.docIdx[x[0]-1][1])
				inexact_idx_elim.append(self.docIdx[x[0]-1][1])
		self.inexact_idx_elim=inexact_idx_elim
	def inexact_query_cluster_pruning(self, query_terms, k):
	#function for exact top K retrieval using cluster pruning (method 4)
	#Returns at the minimum the document names of the top K documents ordered in decreasing order of similarity score
		pass
		current=timer()
		'''
		Normalized query vector
		'''
		q=collections.Counter(query_terms)
		for x,y in q.items():
			#print(self.posting.get(x,[{'IDF':0}])[0]['IDF'])
			q[x]=(1+(math.log10(q[x])))*self.posting.get(x,[{'IDF':0}])[0]['IDF']
		qdist=math.sqrt(sum((q[k])**2 for k in q.keys()))
		if(qdist!=0):
			q={k:v/qdist for k,v in q.items()}
		else:
			q={k:0 for k,v in q.items()}
		
		
		'''
		Cosine similarity of query vector vs the leaders
		'''
		leader_heirarchy=[]
		for d1,tfdf1 in self.leaders.items():
			cos_sim=0
			for q1,tfdf2 in q.items():
				cos_sim+=tfdf1.get(q1,0)*tfdf2
			cos_sim/=self.doclen[d1]
			leader_heirarchy.append((d1,cos_sim))
				
		
		leader_heirarchy=sorted(leader_heirarchy,key=lambda x:-x[1])
		'''
		Cosine similarity of cluster with the leader found vs the query
		'''
		count=k
		result=[]
		cos_cluster={}
		for x1,y1 in leader_heirarchy:
			cos_cluster[x1]=y1
			for x in self.cluster.get(x1,[]):
				cos_sim=0
				for q1,tfdf1 in q.items():
					cos_sim+=tfdf1*self.doc_idf[x].get(q1,0)
				if(cos_sim!=0):
					cos_sim/=self.doclen[x]
					cos_cluster[x]=cos_sim
			result.append(cos_cluster)
			if len(cos_cluster.keys())>=count:
				break
			else:
				count-=len(cos_cluster.keys())
			cos_cluster={}
		
		#print(result)
		end = timer()
		diff = end - current
		self.prune_time=diff
		self.File.write("\n\nInexact cluster pruning search in "+str(diff)+" seconds.\n")
		print("Inexact cluster pruning search in "+str(diff)+" seconds.")
		self.File.write("Result for inexact cluster pruning search for query:"+str(query_terms))
		self.File.write("\n")
		print("Result for inexact cluster pruning search for query:",query_terms)
		
		inexact_cluster_pruning=[]
		c=0
		for z in result:
			for x in sorted(z.items(),key=lambda x:-x[1]):
				if c<k:
					c+=1
					self.File.write(self.docIdx[x[0]-1][1])
					self.File.write("\n")
					print(self.docIdx[x[0]-1][1])
					inexact_cluster_pruning.append(self.docIdx[x[0]-1][1])
				else:
					break
		self.inexact_cluster_pruning=inexact_cluster_pruning
	def print_dict(self):
	#function to print the terms and posting list in the index
		pass
		for k,v in (sorted(self.posting.items(), key=lambda x: x[0])):
			print(k+":[", end="")
			for i in v:
				for x,y in i.items():
					if x!='IDF':
						print("("+str(x)+","+str(y[0])+",",end="")
						for c,w in enumerate(y):
							if c==len(y)-1:
								print("["+str(w),end="")
							elif c!=0:
								print("["+str(w)+",",end="")
						print("])],",end=" ")		
					else:
						print(str(y)+",",end=" ")
	def print_doc_list(self):
	# function to print the documents and their document id
		pass
		for x in self.docIdx:
			print("Doc ID: "+str(x[0])+" ==> "+str(x[1]))
	
	def accuracy(self):
	# returns the accuracy of all the inexact method vs the exact query search
		accuracy={}
		
		accuracy['champ']=(len(set(self.exact)&set(self.inexact_champion))/len(self.exact))*100
		accuracy['index']=(len(set(self.exact)&set(self.inexact_idx_elim))/len(self.exact))*100
		accuracy['prune']=(len(set(self.exact)&set(self.inexact_cluster_pruning))/len(self.exact))*100
		self.acc=accuracy
	def performance(self):
		
		champ=[]
		index=[]
		prune=[]
		exact_time=[]
		champ_time=[]
		index_time=[]
		prune_time=[]
		ns=[]
		k=10
		for x in range(1,100):
			l=random.sample(self.vocab,x)
			with nostdout():
				self.exact_query(l,k)
				self.inexact_query_champion(l,k)	
				self.inexact_query_index_elimination(l,k)
				self.inexact_query_cluster_pruning(l,k)
				self.accuracy()
			champ.append(self.acc['champ'])
			index.append(self.acc['index'])
			prune.append(self.acc['prune'])
			exact_time.append(self.exact_time)
			champ_time.append(self.champ_time)
			index_time.append(self.idx_time)
			prune_time.append(self.prune_time)
			ns.append(x)
		'''
		accuracy measurement
		'''	
		plt.plot(ns, champ, 'g-', label='champion')
		plt.plot(ns, index, 'r-', label='indx elim')
		plt.plot(ns, prune, 'b-', label='cluster prune')
		plt.legend(loc='upper left')
		plt.xlabel('Number of query terms')
		plt.ylabel('Accuracy wrt exact query search')
		plt.suptitle('No of documents selected:'+str(k), fontsize=12)
		plt.show()
		'''
		Time measurement
		'''
		plt.plot(ns, exact_time, 'k-', label='exact')
		plt.plot(ns, champ_time, 'g-', label='champion')
		plt.plot(ns, index_time, 'r-', label='indx elim')
		plt.plot(ns, prune_time, 'b-', label='cluster prune')
		plt.legend(loc='upper left')
		plt.xlabel('Number of query terms')
		plt.ylabel('Time')
		plt.suptitle('No of documents selected:'+str(k), fontsize=12)
		plt.show()
			#print(x)
			#print(self.acc)
		
def main():
	a=index('./collection')
	a.buildIndex()
	
	a.exact_query(['with','without','yemen','yemeni','occasion'],6)
	a.inexact_query_champion(['with','without','yemen','yemeni','occasion'], 6)	
	a.inexact_query_index_elimination(['with','without','yemen','yemeni','occasion'], 6)
	a.inexact_query_cluster_pruning(['with','without','yemen','yemeni','occasion'], 6)
	
	a.exact_query(['meeting','agricultural','experts'],6)
	a.inexact_query_champion(['meeting','agricultural','experts'], 6)	
	a.inexact_query_index_elimination(['meeting','agricultural','experts'], 6)
	a.inexact_query_cluster_pruning(['meeting','agricultural','experts'], 6)
	
	a.performance()
	#a.print_dict()
	#a.print_doc_list()
if __name__ == '__main__':
	main()