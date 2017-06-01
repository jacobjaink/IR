#Python 3.0
import re
import os
import collections
import time
import math
from collections import Counter
from random import randint
import matplotlib.pyplot as plt
import sys

def output():
	orig_stdout = sys.stdout
	f = open('out.txt', 'a')
	sys.stdout = f
	return orig_stdout
#import other modules as needed

class index:
	def __init__(self,path):
		pass
		self.path=path
	def buildIndex(self):
		#function to read documents from collection, tokenize and build the index with tokens
		# implement additional functionality to support relevance feedback
		#use unique document integer IDs
		pass
	
		
		fullfilename = os.path.join(self.path, 'TIME.ALL')	
		#Reading Stop words for filtering
		stopfilename = os.path.join(self.path, 'TIME.STP')
		stopfile=open(stopfilename,'r')
		stopwords=re.sub('\W+',' ',stopfile.read()).lower().split()
		#print(stopwords)
		
		
		self.stopwords=stopwords
		
		'''
		Reading each document one by one from TIME.ALL
		'''
		files = []
		lines=""
		name=''
		documents={}
		with open(fullfilename) as f:
			for line in f:
				if '*TEXT' in line or '*STOP' in line:
					if name!='':
						documents[str(int(name))]=lines
						lines=""
					name=line[6:9]
					if name!='':
						files.append(str(int(name)))
				elif len(line.strip())!=0:
					#print(line)
					#line=line.strip('\n')
					line=line.replace('\n',' ')
					lines=lines+line
		#print(documents)
		
		
		vocab={}
		docId={}
		docList=[]
		pos_idx={}
		#print(len(files))
		#print(files)
		for x,y in enumerate(files):
			#print(y)
			docId[y]=x+1
			docList.append((x+1,y))
		'''
		creating index with term frequency and default IDF as zero
		'''
		for doc,content in documents.items():
			tokens=list(re.findall('[a-z]+', content.lower()))
			tokens=[x for x in tokens if x not in stopwords]
			did=docId[doc]
			for i,t in enumerate(tokens):
				dic={}
				if t not in vocab:
					vocab[t]=[did]
					dic[did]=[1]
					dic[did].append(i+1)
					pos_idx[t]=[{'IDF':0}]
					pos_idx[t].append(dic)
				elif did not in vocab[t]:
					vocab[t].append(did)
					dic[did]=[1]
					dic[did].append(i+1)
					pos_idx[t].append(dic)
				else:					
					for x in pos_idx[t]:
						if did in x:
							x[did][0]+=1
							x[did].append(i+1)	
		#print(files)
		#print(len(vocab),sorted(vocab))
		
		self.vocab=vocab.keys()
		'''
		setting the IDF value and the weighted term frequency
		'''
		
		for k,v in pos_idx.items():
			v[0]['IDF']=math.log10(len(files)/(len(v)-1))
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
			v[0]['IDF']=math.log10(len(files)/(len(v)-1))
			avg_len+=len(v)-1
			for i in v:
				for x,y in i.items():
					if x!='IDF':
						doc_length[x]=doc_length.get(x,0)+(y[0]*v[0]['IDF'])**2
		
		#print(doc_length)
		avg_len/=len(pos_idx)
		#print(round(avg_len))
		doc_length={k:math.sqrt(v) for k,v in doc_length.items()}
		
		#print(docId)
		#print(doc_length[423])
		#print(len(did),sorted(did.items(),key= lambda x:-x[1]))
		
		
		'''
		vector model for each document with tf-idf as the vector magnitude
		'''
		doc_idf={}		
		for key,v in pos_idx.items():
			for x in v:
				for k1,v1 in x.items():
					if k1!='IDF':
						if k1 not in doc_idf:
							tfdf=(v1[0]*v[0]['IDF'])/doc_length[k1]
							doc_idf[k1]={key:tfdf}
						else:
							tfdf=(v1[0]*v[0]['IDF'])/doc_length[k1]
							doc_idf[k1][key]=tfdf
	
		#print(doc_idf[423])
		#print(doc_length)					
													
		#print(len(doc_idf[423]))
		#print(docId['TEXT 563'])
		self.doc_vector=doc_idf
		self.posting=pos_idx
		self.docIdx=docList
		self.doclen=doc_length
		self.docmap=docId
		'''
		print(sorted(doc_idf.keys()))
		print(sorted(docId.values()))
		print(sorted(list(map(int,docId.keys()))))
		print(docList)
		#print(docList)
		'''
	def rocchio(self, query_terms, pos_feedback, neg_feedback, alpha, beta, gamma,query_vector={}):
	#function to implement rocchio algorithm
	#pos_feedback - documents deemed to be relevant by the user
	#neg_feedback - documents deemed to be non-relevant by the user
	#Return the new query  terms and their weights
		pass
		if len(query_vector)==0:
			'''
			compute vector form of query
			'''
			q=collections.Counter(query_terms)
			for x,y in q.items():
				#print(self.posting.get(x,[{'IDF':0}])[0]['IDF'])
				q[x]=(1+(math.log10(q[x])))*self.posting.get(x,[{'IDF':0}])[0]['IDF']
		else:
			q=query_vector
		qdist=math.sqrt(sum((q[k])**2 for k in q.keys()))
		'''
		normalizing the query vector and multiplying by alpha
		'''
		if(qdist!=0):
			q={k:((v/qdist)*alpha) for k,v in q.items()}
		else:
			q={k:0 for k,v in q.items()}
		
		#print(q)
		'''
		Adding all the vectors of positive document and length normalizing
		'''
		pos=collections.Counter()
		if(len(pos_feedback)!=0):
			for x in pos_feedback:
				pos=pos+Counter(self.doc_vector[int(self.docmap[x])])
			
			pos_norm=beta/len(pos_feedback)	
			pos={k:(v*(pos_norm)) for k,v in pos.items()}
		
		'''
		Adding all the vectors of negative document and length normalizing
		'''
		neg=collections.Counter()
		if(len(neg_feedback)!=0):
			for x in  neg_feedback:
				neg=neg+Counter(self.doc_vector[int(self.docmap[x])])
			
			neg_norm=gamma/len(neg_feedback)		
			neg={k:(v*(neg_norm)) for k,v in neg.items()}
		
		qopt=collections.Counter()
		
		
		'''
		Optimized query vector
		'''
		qopt=Counter(q)+Counter(pos)-Counter(neg)
		'''
		Removing terms with negative weights
		'''
		qopt={k:v for k,v in qopt.items() if v>0}
		
		'''
		retains all the original query terms and adds the highly weighted 10 words extra to query
		Found its not so efficient as the Rocchio algorithm performance is reduced considerably
		
		count=0
		newq={}
		for x, y in sorted(qopt.items(),key=lambda x:-x[1]):
			if x in q:
				newq[x]=y
			elif count<=10:
				newq[x]=y
				count+=1
			
		
		print(newq)
		'''
		return qopt
	
	def query(self, query_terms, k, qopt={}):
	#function for exact top K retrieval using cosine similarity
	#Returns at the minimum the document names of the top K documents ordered in decreasing order of similarity score
		pass
		if len(qopt)==0:
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
		else:
			q=qopt	
		'''
		calculating the document score with respect to each query
		'''
		#print(q)
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
		#print(doc_score)
		
		'''
		Displaying the K list of documents in decreasing document score order
		'''
		exact=[]
		exact_doc=[]
		s=output()	
		print('Query Vector: '+str(query_terms))
		sys.stdout = s
		print('The top '+str(k)+' documents for the query are:')
		for count,x in enumerate(doc_score):
			if(count<k):
				exact.append(str(x[0]))
				exact_doc.append(str(self.docIdx[x[0]-1][1]))
				print("Doc=>"+str(self.docIdx[x[0]-1][1])+" DocId=>" + str(x[0]))
		self.exact=exact
		self.exact_doc=exact_doc
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
	
	def exp_study(self):
		fullfilename = os.path.join(self.path, 'TIME.QUE')
		
		queries={}
		name=''
		lines=""
		
		'''
		Reading the TIME.QUE document
		'''
		with open(fullfilename) as f:
			for line in f:
				if '*FIND' in line or '*STOP' in line:
					if name!='':
						tokens=list(re.findall('[a-z]+', lines.lower()))
						tokens=[x for x in tokens if x not in self.stopwords]
						queries[str(int(name.strip()))]=tokens
						lines=""
					name=(line[5:]).strip()
				elif len(line.strip())!=0:
					line=line.replace('\n',' ')
					lines=lines+line
		#print(queries)
		'''
		Reading TIME.REL document
		'''
		fullfilename = os.path.join(self.path, 'TIME.REL')
		rel={}
		with open(fullfilename) as f:
			for line in f:
				if len(line.strip())!=0:
					line=line.split()
					rel[line[0]]=line[1:]
		
		#print(rel)
		
		# 3 queries fixed for experimental study
		No_of_queries=3
		len_given=5
		selected=[]
		sel=-1
		pre=[]
		rec=[]
		mapr=[]
		qopt={}
		ns=[x for x in range(1,6)]
		ret_cnt=[]
		queries_used=[]
		for y in range(No_of_queries):
			s=output()
			print('\nQuery '+str(y+1)+' terms and weights in each iteration')
			print('\n')
			sys.stdout = s
			p=[]
			r=[]
			m=[]
			k=0
			for x in range(5):
				if x==0:
					rel_len=0
					'''
					Random selection of query
					'''
					while(rel_len<5):
						sel=str(randint(1,len(rel)))
						if sel not in selected:
							rel_len=len(rel[str(sel)])
						else:
							rel_len=0
						selected.append(sel)
					queries_used.append(sel)
					#print(sel)
					if rel_len<10:
						k=20
					elif rel_len<15:
						k=30
					elif rel_len<20:
						k=40
					else:
						k=2*rel_len
					ret_cnt.append(k)
					print("\n\nFind "+sel+" query used for analysis:"+str(queries[sel]))
					#print(queries[sel])
					self.query(queries[sel],k)
					pr,rc,ma=self.prm(self.exact,rel[sel])
					p.append(pr)
					r.append(rc)
					m.append(ma)
					pos=input('\nEnter positive feedback documents for next query(if any space separated, doc number(not id):')
					neg=input('Enter negative feedback documents for next query(if any space separated, doc number(not id)):')
					qopt=self.rocchio(queries[sel], pos.split(), neg.split(), 1, 0.75, 0.15)
				else:
					self.query(qopt,k,qopt)
					pr,rc,ma=self.prm(self.exact,rel[sel])
					p.append(pr)
					r.append(rc)
					m.append(ma)
					if(x<4):
						pos=input('Enter positive feedback documents(if any space separated, doc number(not id)):')
						neg=input('Enter negative feedback documents(if any space separated, doc number(not id)):')
						qopt=self.rocchio(qopt.keys(), pos.split(), neg.split(), 1, 0.75, 0.15,qopt)
			print("\nQuery "+str(y+1))
			print("Precision"+str(p))
			print("Recall"+str(r))
			print("MAP"+str(m))
			pre.append(p)
			rec.append(r)
			mapr.append(m)
		
		color=['g-','r-','b-']
		plt.ion()
		'''
		plotting all the graph
		'''
		for i in range(No_of_queries):
			plt.plot(ns, pre[i], color[i], label='query'+str(i+1))
		plt.legend(loc='upper left')
		plt.xlabel('Rocchio Iterations')
		plt.ylabel('Precision wrt TIME.REL')
		plt.suptitle('No of documents selected:'+str(ret_cnt), fontsize=12)
		plt.savefig('fig1')
		plt.close()
		
		for i in range(No_of_queries):
			plt.plot(ns, rec[i], color[i], label='query'+str(i+1))
		plt.legend(loc='upper left')
		plt.xlabel('Rocchio Iterations')
		plt.ylabel('Recall wrt TIME.REL')
		plt.suptitle('No of documents selected:'+str(ret_cnt), fontsize=12)
		plt.savefig('fig2')
		plt.close()
		
		for i in range(No_of_queries):
			plt.plot(ns, mapr[i], color[i], label='query'+str(i+1))
		plt.legend(loc='upper left')
		plt.xlabel('Rocchio Iterations')
		plt.ylabel('MAP wrt TIME.REL')
		plt.suptitle('No of documents selected:'+str(ret_cnt), fontsize=12)
		plt.savefig('fig3')
		plt.close()
		
		
		'''
		pseudo relevance feedback
		'''
		ppre=[]
		prec=[]
		pmapr=[]
		for cnt,y in enumerate(queries_used):
			s=output()
			sel=y
			print('\nQuery '+str(cnt+1)+' terms and weights in each iteration for pseudo relevance')
			print('\n')
			sys.stdout = s
			p=[]
			r=[]
			m=[]
			k=ret_cnt[cnt]
			for x in range(5):
				if x==0:	
					print("\n\nFind "+sel+" query used for analysis:"+str(queries[sel]))
					self.query(queries[sel],k)
					pr,rc,ma=self.prm(self.exact,rel[sel])
					p.append(pr)
					r.append(rc)
					m.append(ma)
					qopt=self.rocchio(queries[sel], self.exact_doc[0:3], [], 1, 0.75, 0.15)
				else:
					self.query(qopt,k,qopt)
					pr,rc,ma=self.prm(self.exact,rel[sel])
					p.append(pr)
					r.append(rc)
					m.append(ma)
					if(x<4):
						qopt=self.rocchio(qopt.keys(), self.exact_doc[0:3], [], 1, 0.75, 0.15,qopt)
			print("\nQuery "+str(cnt+1))
			print("Precision"+str(p))
			print("Recall"+str(r))
			print("MAP"+str(m))
			ppre.append(p)
			prec.append(r)
			pmapr.append(m)
		
		for i in range(No_of_queries):
			plt.plot(ns, pre[i], 'g-', label='user query'+str(i+1))
			plt.plot(ns, ppre[i], 'r-', label='pseudo query'+str(i+1))
			plt.legend(loc='upper left')
			plt.xlabel('Rocchio Iterations')
			plt.ylabel('Precision wrt TIME.REL')
			plt.suptitle('No of documents selected:'+str(ret_cnt[i]), fontsize=12)
			plt.savefig('pseudoprefig'+str(i+1))
			plt.close()
		
		for i in range(No_of_queries):
			plt.plot(ns, rec[i], 'g-', label='user query'+str(i+1))
			plt.plot(ns, prec[i], 'r-', label='pseudo query'+str(i+1))
			plt.legend(loc='upper left')
			plt.xlabel('Rocchio Iterations')
			plt.ylabel('Recall wrt TIME.REL')
			plt.suptitle('No of documents selected:'+str(ret_cnt[i]), fontsize=12)
			plt.savefig('pseudorecfig'+str(i+1))
			plt.close()
		
		for i in range(No_of_queries):
			plt.plot(ns, mapr[i], 'g-', label='user query'+str(i+1))
			plt.plot(ns, pmapr[i], 'r-', label='pseudo query'+str(i+1))
			plt.legend(loc='upper left')
			plt.xlabel('Rocchio Iterations')
			plt.ylabel('MAP wrt TIME.REL')
			plt.suptitle('No of documents selected:'+str(ret_cnt[i]), fontsize=12)
			plt.savefig('pseudomapfig'+str(i+1))
			plt.close()
	#Calculate the precision, recall and MAP for the two list of documents provided
	def prm(self,result,relevant):
		
		precision=(len(set(result)&set(relevant))/len(result))
		recall=(len(set(result)&set(relevant))/len(relevant))
		mapr=0.0
		
		rel=0
		for x,y in enumerate(result):
			if y in relevant:
				rel+=1
				mapr=mapr+(rel/(x+1))
		
		mapr/=len(relevant)
		return precision, recall, mapr
		
def main():
	
	a=index('./time')
	a.buildIndex()
	print('Simple test\nQuery:'+str(['plummeted','national','assasinate','rose','nature']))
	a.query(['plummeted','national','assasinate','rose','nature'], 6)
	pos=input('Enter positive feedback documents(if any space separated):')
	neg=input('Enter negative feedback documents(if any space separated):')
	qopt=a.rocchio(['plummeted','national','assasinate','rose','nature'], pos.split(), neg.split(), 1, 0.75, 0.15)
	print('Output after Rocchio algorithm is applied:')
	a.query(qopt.keys(), 6)
	a.exp_study()
	
	
if __name__ == '__main__':
	main()