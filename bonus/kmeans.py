# Python 3.0
import re
import os
import collections
from collections import Counter
import time
import tarfile
import math
import random
import matplotlib.pyplot as plt
import contextlib
import sys
from networkx.algorithms import cluster
import numpy

@contextlib.contextmanager
def nostdout():
	save_stdout = sys.stdout
	sys.stdout = open('trash','w')
	yield
	sys.stdout = save_stdout
# import other modules as needed
# implement other functions as needed
class kmeans:
	
	def __init__(self, path_to_collection):
		tar = tarfile.open(path_to_collection)
		tar.extractall()
		tar.close()
	def buildIndex(self):
		#function to read documents from collection, tokenize and build the index with tokens
		# implement additional functionality to support relevance feedback
		#use unique document integer IDs
		pass
	
		
		fullfilename = 'TIME.ALL'	
		#Reading Stop words for filtering
		stopfilename = 'TIME.STP'
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
		
		
		avg_len/=len(pos_idx)
		doc_length={k:math.sqrt(v) for k,v in doc_length.items()}
		
		
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
	
		#print(doc_idf)
		self.doc_vector=doc_idf
		#self.posting=pos_idx
		self.docIdx=docList
		self.doclen=doc_length
		self.docmap=docId
		
		#print(sorted(doc_idf.items()))
		'''
		print(sorted(docId.values()))
		print(sorted(list(map(int,docId.keys()))))
		print(docList)
		#print(docList)
		'''
	def clustering(self, kvalue):
	# function to implement kmeans clustering algorithm
	# Print out:
	# #For each cluster, its RSS values and the document ID of the document closest to its centroid.
    # #Average RSS value
	# #Time taken for computation.
		pass
		
		'''
		kmeans ++ random initialization of data points based on probability of distance from the closest centroid already found
		'''
		l=[x for x in range(1,len(self.docIdx)+1)]
		clusters=[]
		clusters.append(random.sample(l,1)[0])
		
		y=0
		dist_prob={}
		tot_dist=0.0
		tot=0
		#print(clusters)
		while(len(clusters)!=kvalue):
			points=[]
			probs=[]
			for k,v in self.doc_vector.items():
				v1=self.doc_vector[clusters[y]]
				val1=int(math.sqrt(sum((v.get(d,0.0) - v1.get(d,0.0))**2 for d in set(v) | set(v1))))
				val2=dist_prob.get(k,sys.maxsize)
				val3=min(val1,val2)
				dist_prob[k]=val3
				if val3!=val2 and val2!=sys.maxsize:
					tot_dist=tot_dist-val2+val3
				elif val1!=val2 and val2!=val3:
					tot_dist+=val3
				tot=0
				for k,v in dist_prob.items():
					tot+=v
				#print(val1,val2,val3,tot_dist,tot)
			for k1,v1 in dist_prob.items():
				points.append(k1)
				probs.append(v1/tot_dist)		
			
			s=clusters[y]
			while(s in clusters):
				s=numpy.random.choice(points,1, p=probs)[0]
			clusters.append(s)
			y+=1
			#print(clusters)
		
		#print(self.docIdx)
		clusters=[x[0] for c,x in enumerate(self.docIdx) if c+1 in clusters]
		centroids={k:v for k,v in self.doc_vector.items() if k in clusters}
		#print(len(centroids))
		#print(centroids.keys())
		eucl=[]
		clustering={}
		doc_cluster={}
		doc_cluster_old={}
		state=True
		#print(centroids)
		i=0
		#print(self.doc_vector)
		while(state):
			for k,v in self.doc_vector.items():
				eucl=[]
				for c1,v1 in centroids.items():
					#eucl.append((c1,math.sqrt(sum((v.get(d,0.0) - v1.get(d,0.0))**2 for d in set(v) | set(v1)))))
					'''
					length normalizing the centroid and document vector
					
					cdist=math.sqrt(sum((v1[k])**2 for k in v1.keys()))
					if(cdist!=0):
						v1norm={k:v/cdist for k,v in v1.items()}
					else:
						v1norm={k:0 for k,v in v1.items()}
						
					vdist=self.doclen[k]
					if(vdist!=0):
						vnorm={k:v/vdist for k,v in v.items()}
					else:
						vnorm={k:0 for k,v in v.items()}
					'''
					
					'''
					calculating cosine similarity
					'''
					cos_sim=0.0	
					for dkey,dtfdf in v.items():
						cos_sim+=dtfdf*v1.get(dkey,0.0)
					eucl.append((c1,cos_sim))
				#print(eucl)
				eucl=sorted(eucl,key=lambda x:-x[1])
				#print(self.docIdx[eucl[0][0]-1],eucl[0][0])
				
				#print(eucl)
				'''
				Re-organizing documents in each iteration as the centroid gets recentered
				'''
				if i!=0:
					clustering[doc_cluster[k]].remove(k)
				if eucl[0][0] in clustering:
					clustering[eucl[0][0]].append(k)
				else:
					clustering[eucl[0][0]]=[k]
				doc_cluster[k]=eucl[0][0]
			
			#print(clustering)	
			'''
			recenter the centroid based on the cluster formed by calculating mean vector of cluster points
			'''
			a=Counter()
			for k in centroids.keys():
				a=Counter()
				for i in clustering[k]:
					a+=Counter(self.doc_vector[k])
				if(len(a)!=0):
					centroids[k]=dict(a)
			
			for k,v in centroids.items():
				l=len(clustering[k])
				if l!=0:
					for k1,v1 in v.items():
						centroids[k][k1]/=l
			for k,v in centroids.items():
				cdist=math.sqrt(sum((centroids[k][k1])**2 for k1 in centroids[k].keys()))
				if(cdist!=0):
					centroids[k]={k1:v1/cdist for k1,v1 in centroids[k].items()}
				else:
					centroids[k]={k1:0 for k1,v1 in centroids[k].items()}
			'''
			convergence rule - documents don't get reorganized to new cluster
			'''			
			if doc_cluster==doc_cluster_old:
				state=False
			doc_cluster_old=doc_cluster
			i+=1
		print("Clustering generated:")
		print(clustering)
		
		'''
		RSS calculation
		'''
		RSS={}
		l=[]
		closest={}
		for x,y in clustering.items():
			RSS[x]=0
			if len(y)!=0:
				l=[]
				for z in y:
					p=sum((self.doc_vector[z].get(d,0.0) - centroids[x].get(d,0.0))**2 for d in set(self.doc_vector[z]) | set(centroids[x]))
					RSS[x]+=p
					l.append((z,p))
				closest[x]=sorted(l,key=lambda x:x[1])[0][0]
		
		print("RSS for each cluster:")
		print(RSS)
		print("Point closer to each cluster:")
		print(closest)
		
		avg=0.0
		for k,v in RSS.items():
			avg+=v
		print("average RSS:")
		print(avg/kvalue)
		
		return avg
			
def main():
	a = kmeans("time.tar.gz")
	current = time.time()
	a.buildIndex()
	a.clustering(10)
	end = time.time()
	diff = end - current
	print("Time Taken:",diff)
	
	ns=[]
	rss=[]
	with nostdout():
		for i in range(2,31):
			ns.append(i)
			rss.append(a.clustering(i))
	
	plt.plot(ns, rss, 'g-', label='RSS')
	plt.legend(loc='upper left')
	plt.xlabel('Number of centroids(K)')
	plt.ylabel('RSS')
	plt.suptitle('RSS vs K', fontsize=12)
	plt.savefig('RSS')
	
if __name__ == '__main__':
	main()
	
