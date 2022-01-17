from models import *
from helper import *
from random import *
from pprint import pprint
import pandas as pd
import scipy.sparse as sp
import uuid, sys, os, time, argparse
import pickle, pdb, operator, random, sys
import tensorflow as tf
from collections import defaultdict as ddict
from sklearn.metrics import precision_recall_fscore_support


YEARMIN = -50 # YAGO의 최소 year는 -430인데?
YEARMAX = 3000
class HyTE(Model):
	def read_valid(self,filename): # onlyTest가 parameter로 들어오면 valid.txt와 test.txt를 읽어서 (s, r, t, (start, end))의 쿼드러플로 만들어주는 함수
		valid_triples = []
		with open(filename,'r') as filein:
			temp = []
			for line in filein:
				temp = [int(x.strip()) for x in line.split()[0:3]]
				temp.append([line.split()[3],line.split()[4]])
				valid_triples.append(temp)
		return valid_triples

	def getOneHot(self, start_data, end_data, num_class):
		temp = np.zeros((len(start_data), num_class), np.float32)
		for i, ele in enumerate(start_data):
			if end_data[i] >= start_data[i]:
				temp[i,start_data[i]:end_data[i]+1] = 1/(end_data[i]+1-start_data[i]) 
			else:
				pdb.set_trace()
		return temp	



	def getBatches(self, data, shuffle = True):
		if shuffle: random.shuffle(data)
		num_batches = len(data) // self.p.batch_size

		for i in range(num_batches):
			start_idx = i * self.p.batch_size
			yield data[start_idx : start_idx + self.p.batch_size]


	def create_year2id(self,triple_time): # load_data()에서만 쓰임
        """
        triple_time의 key는 train.txt에서의 index이고
        value는 time_interval [start, end]이다.
        이때 start와 end는 year part만 남아있다.
        
        input: triple_time = dict{index: [start, end], ...}
        output: self.year_list, year2id
        각각 self.year_list는 start와 end에 등장했던적이 있는 중복허용 year
        year2id는 300개씩 끊어서 만든 year class의 id (time step)
        """
		year2id = dict()
		freq = ddict(int)
		count = 0
		year_list = []

		for k,v in triple_time.items(): # items generate key, value pair tuple
			try:
				start = v[0].split('-')[0] # split을 사용할 필요가 없음
				end = v[1].split('-')[0]
			except:
				pdb.set_trace() # debugger

			if start.find('#') == -1 and len(start) == 4: year_list.append(int(start))
            # find: 있으면 0, 없으면 -1
            # start에 '#'이 없고 and start의 len이 4 (백단위 년도는 세지 않는다.)
	    # 왜 안세냐고 대체
			if end.find('#') == -1 and len(end) ==4: year_list.append(int(end))
		
		year_list.sort()
        # 지금까지 등장했던 fact들의 4자리 년도들의 집합 (start, end 상관없음)
		for year in year_list:
			freq[year] = freq[year] + 1
        # 이 년도들의 개수를 세어준다.

		year_class =[]
		count = 0
		for key in sorted(freq.keys()): # freq의 key들은 이미 sorting이 되어있는데..
			count += freq[key]
			if count > 300:
				year_class.append(key) # 딱 넘기는 순간의 year가 append 된다.
				count = 0
		prev_year = 0
		i=0
		for i,yr in enumerate(year_class): # index, 그리고 갈리는 기준점이 되는 year들
			year2id[(prev_year,yr)] = i 
            # key: 0~100, 101~200, 201~1400, ....
            # val: 0    , 1,       2,       ....
			prev_year = yr+1
		year2id[(prev_year, max(year_list))] = i + 1 
        # 마지막 year_list는 count 300을 못넘겼을 수도 있어서 해주는걸로 보임
		
        self.year_list =year_list		
		return year2id
	def get_span_ids(self, start, end):
		start =int(start)
		end=int(end)
		if start > end:
			end = YEARMAX

		if start == YEARMIN:
			start_lbl = 0
		else:
			for key,lbl in sorted(self.year2id.items(), key=lambda x:x[1]):
				if start >= key[0] and start <= key[1]:
					start_lbl = lbl
		
		if end == YEARMAX:
			end_lbl = len(self.year2id.keys())-1
		else:
			for key,lbl in sorted(self.year2id.items(), key=lambda x:x[1]):
				if end >= key[0] and end <= key[1]:
					end_lbl = lbl
		return start_lbl, end_lbl

	def create_id_labels(self,triple_time,dtype):
        """
        load data에서만 사용됨
        triple_time = dict{index: [start, end], ...}
        input: triple_time, dtype ('triple'이 들어왔음)
        """
		YEARMAX = 3000
		YEARMIN =  -50
		
		inp_idx, start_idx, end_idx =[], [], []
		
		for k,v in triple_time.items():
			start = v[0].split('-')[0] # 얘도 split 필요 없음
			end = v[1].split('-')[0]
			if start == '####':
				start = YEARMIN
			elif start.find('#') != -1 or len(start)!=4:
                # 그냥 == 0 할것이지
                # start에 #이 있거나, start의 len이 4가 아니라면 continue
                # ####이 아니면 #이 있는 경우가 없음
                # 이건 triple_time만들때 parsing 잘못해서
                # "" <- 이게 start로 들어오거나 100같이 세자리수 year가 들어오는 경우임
                # 아니 도대체 세자리수는 왜 passing/filtering하는거임?
				continue # 해당 iteration을 건너뜀

			if end == '####':
				end = YEARMAX
			elif end.find('#')!= -1 or len(end)!=4:
				continue # end에 #이 있거나, len이 4가 아니라면 iteration 건너뜀
			
			start = int(start) # '#'이  들어올 일은 없다.
			end = int(end)
			
			if start > end: # 와 이런경우 진짜 있어 데이터에 Ex) (9738 9 9739 2013-##-##	2004-##-##)
				end = YEARMAX
			inp_idx.append(k) # 아니 하 진짜 왜 이딴곳에다가..
            # continue로부터 살아남은 놈들의 index이다.

			if start == YEARMIN: # only if start == "####" 근데 -405도 있는데 왜 -50이 mean이고 index 0이냐?
				start_idx.append(0)
			else:
				for key,lbl in sorted(self.year2id.items(), key=lambda x:x[1]): # ..sorting되어있는걸 또하고 또하고..
					# year2id를 쭈욱 scan하면서 trainset으로 만든 triple_time의 순서에 맞는 id들을
                    # start_idx에 넣고있다. 즉, train_set의 time interval들을 id화 하고있음
                    if start >= key[0] and start <= key[1]:  # key = (prev, yr), prev < end < yr인 상황
						start_idx.append(lbl)
            """
            start_idx는 continue에 걸리지 않는 이상 무조건 train set갯수만큼 생성됨
            """
			
			if end == YEARMAX: # 끝까지 ( end time이 ####인 경우이다.)
				end_idx.append(len(self.year2id.keys())-1)
                # end_idx가 처음 등장했다
                # 전체 fact에 등장한 year들의 갯수 -1 개를 저장하는데 뭐하자는건진 모르겠음
			else:
				for key,lbl in sorted(self.year2id.items(), key=lambda x:x[1]):
					if end >= key[0] and end <= key[1]: # key = (prev, yr), prev < end < yr인 상황
						end_idx.append(lbl)

		return inp_idx, start_idx, end_idx 
        # index, start가 어디 timestep에 속하는지, end가 어디 timestep에 속하는지
	
	def load_data(self):
		triple_set = []
		with open(self.p.triple2id,'r') as filein:
			for line in filein:
                # strip: white space 제거
				tup = (int(line.split()[0].strip()) , int(line.split()[1].strip()), int(line.split()[2].strip()))
				triple_set.append(tup)
		triple_set=set(triple_set)

		train_triples = []
        # ddict란 deafualt dictionary이며, 그냥 dictionary인데 인자로 받은 datatype으로 초기화가 된다.
        # ddict(dict)는 안에 value가 dictionary들이 들어간다는 뜻
		self.start_time , self.end_time, self.num_class  = ddict(dict), ddict(dict), ddict(dict)
		triple_time, entity_time = dict(), dict()
		self.inp_idx, self.start_idx, self.end_idx ,self.labels = ddict(list), ddict(list), ddict(list), ddict(list)
		max_ent, max_rel, count = 0, 0, 0

		with open(self.p.dataset,'r') as filein: # dataset은 train.txt를 말한다
			for line in filein:
				train_triples.append([int(x.strip()) for x in line.split()[0:3]]) # 각 fact의 [s, r, o]가 append됨
				triple_time[count] = [x.split('-')[0] for x in line.split()[3:5]] 
                # 각 start, end의 숫자부분을 가지고 오려고 했던거 같은데..
                # 이러면 -405는 뭐가됨? -> github에 report했음
				count+=1

		with open(self.p.entity2id,'r', encoding="utf-8") as filein2:
			for line in filein2:
				max_ent = max_ent+1 # entity 개수를 세는듯 하다.
        print("max_ent_num: ", max_ent)

		self.year2id = self.create_year2id(triple_time) # output: self.year_list, year2id
		self.inp_idx['triple'], self.start_idx['triple'], self.end_idx['triple'] = self.create_id_labels(triple_time,'triple')	
		# index, start가 어디 timestep에 속하는지, end가 어디 timestep에 속하는지

        # 이 for문은 돌지 않는다.
        # self.inp_idx['entity']는 선언된적이 없기 때문
        # 중요한 코드는 아닌걸로 보이니 넘어가자
        for i,ele in enumerate(self.inp_idx['entity']):
			if self.start_idx['entity'][i] > self.end_idx['entity'][i]:
				print(self.inp_idx['entity'][i],self.start_idx['entity'][i],self.end_idx['entity'][i])
		self.num_class = len(self.year2id.keys())
        # (prev, yr) pair의 갯수이다.
		
		keep_idx = set(self.inp_idx['triple'])
        # 그냥 index이다. set([range(index num)])이랑 똑같음

		for i in range (len(train_triples)-1,-1,-1): # range(start, stop, step)
			if i not in keep_idx:
				del train_triples[i] 
                # start / end time이 -405이런거나 100 이런경우 삭제해버린다.
                # 왜 니 맘대로 삭제하는데?

		with open(self.p.relation2id, 'r') as filein3:
			for line in filein3:
				max_rel = max_rel +1
                # relation 개수를 세는듯 하다
        print("Relation num: ", max_rel)
		index = randint(1,len(train_triples))-1
        # 범위내에서 랜덤하게 하나는 뽑아서 1을 뺀다
		
		posh, rela, post = zip(*train_triples)
		head, rel, tail = zip(*train_triples)
        # train triples를 s, r, o로 나눠줬다.
        # pos는 positive가 아닐까 생각함
        # rela의 a는 뭔지 모르겠음

		posh = list(posh) 
		post = list(post)
		rela = list(rela)

		head  =  list(head) 
		tail  =  list(tail)
		rel   =  list(rel)

		for i in range(len(posh)): # 말이 len(posh)이지, 그냥 train_triple 수만큼 반복
			if self.start_idx['triple'][i] < self.end_idx['triple'][i]: # start랑 end랑 다른 year class에 속하는 경우
				for j in range(self.start_idx['triple'][i] + 1,self.end_idx['triple'][i] + 1):
                    #start yearclass +1 부터 end yearclass까지
					head.append(posh[i])
					rel.append(rela[i])
					tail.append(post[i])
                    # 멀쩡한 head, rel, tail에다가 속하는 yearclass 수만큼 append 해준다..?
					self.start_idx['triple'].append(j)
                    # 속하는 yearclass를 전부 다 넣어주는중

		self.ph, self.pt, self.r,self.nh, self.nt , self.triple_time  = [], [], [], [], [], []
		for triple in range(len(head)): # 기존 triple에다가 뒤에 yearclass만큼 반복시켜놓은놈들 추가
			neg_set = set()
			for k in range(self.p.M): # M은 neg_sample이다. default 5
				possible_head = randint(0,max_ent-1) # head entity로 쓸 entity하나 뽑기
				while (possible_head, rel[triple], tail[triple]) in triple_set or (possible_head, rel[triple],tail[triple]) in neg_set:
					possible_head = randint(0,max_ent-1)
                    #unique한 negset을 만들고 싶음
				self.nh.append(possible_head) # 틀린 머리 (negativ head)
				self.nt.append(tail[triple])  # 정답 tail
				self.r.append(rel[triple])    # 정답 relation
				self.ph.append(head[triple])  # 정답 head
				self.pt.append(tail[triple])  # 정답 tail # 그냥 가져다 써 왜 nt pt 나눠놓은거야 대체 왜
				self.triple_time.append(self.start_idx['triple'][triple]) # 내가 속한 yearclass
				neg_set.add((possible_head, rel[triple],tail[triple])) \
                # negative triple (이 set은 그냥 negative triple이 unique한지를 검사하려고 만든거 같음)
		
		for triple in range(len(tail)):
			neg_set = set()
			for k in range(self.p.M):
				possible_tail = randint(0,max_ent-1)
				while (head[triple], rel[triple],possible_tail) in triple_set or (head[triple], rel[triple],possible_tail) in neg_set:
					possible_tail = randint(0,max_ent-1)
				self.nh.append(head[triple])
				self.nt.append(possible_tail)
				self.r.append(rel[triple])
				self.ph.append(head[triple])
				self.pt.append(tail[triple])
				self.triple_time.append(self.start_idx['triple'][triple])
				neg_set.add((head[triple], rel[triple],possible_tail))
        # tail도 같은 과정 거친다.

		self.max_rel = max_rel # total relation수 (train + test + valid)
		self.max_ent = max_ent # total entity 수 (train + test + valid)
		self.max_time = len(self.year2id.keys()) # year class 수
		self.data = list(zip(self.ph, self.pt, self.r , self.nh, self.nt, self.triple_time))
        # pos head, pos tail, rel, neg head, neg tail인데
        # neg head가 pos head랑 같으면 neg tail이 negative이고
        # neg tail이 pos tail이랑 같으면 neg head가 negative임
		self.data = self.data + self.data[0:self.p.batch_size]
        # 그리고 마지막 의문의 복붙 0~ 50000까지를 복사해서 끝에다가 덧붙인다.
        # batch size default = 50000


	def calculated_score_for_positive_elements(self, t, epoch, f_valid, eval_mode='valid'):
		loss =np.zeros(self.max_ent)
		start_trip 	= t[3][0].split('-')[0]
		end_trip 	= t[3][1].split('-')[0]
		if start_trip == '####':
			start_trip = YEARMIN
		elif start_trip.find('#') != -1 or len(start_trip)!=4:
			return

		if end_trip == '####':
			end_trip = YEARMAX
		elif end_trip.find('#')!= -1 or len(end_trip)!=4:
			return
			
		start_lbl, end_lbl = self.get_span_ids(start_trip, end_trip)
		if eval_mode == 'test':
			f_valid.write(str(t[0])+'\t'+str(t[1])+'\t'+str(t[2])+'\n')
		elif eval_mode == 'valid' and epoch == self.p.test_freq:
			f_valid.write(str(t[0])+'\t'+str(t[1])+'\t'+str(t[2])+'\n')

		pos_head = sess.run(self.pos ,feed_dict = { self.pos_head:  	np.array([t[0]]).reshape(-1,1), 
												   	self.rel:       	np.array([t[1]]).reshape(-1,1), 
												   	self.pos_tail:	np.array([t[2]]).reshape(-1,1),
												   	self.start_year :np.array([start_lbl]*self.max_ent),
												   	self.end_year : np.array([end_lbl]*self.max_ent),
												   	self.mode: 			   -1,
												   	self.pred_mode: 1,
												   	self.query_mode: 1})
		pos_head = np.squeeze(pos_head)
		
		pos_tail = sess.run(self.pos ,feed_dict = {    self.pos_head:  	np.array([t[0]]).reshape(-1,1), 
													   self.rel:       	np.array([t[1]]).reshape(-1,1), 
													   self.pos_tail:	np.array([t[2]]).reshape(-1,1),
													   self.start_year :np.array([start_lbl]*self.max_ent),
													   self.end_year : np.array([end_lbl]*self.max_ent),
													   self.mode: 			   -1, 
													   self.pred_mode:  -1,
													   self.query_mode:  1})
		pos_tail = np.squeeze(pos_tail)


		pos_rel = sess.run(self.pos ,feed_dict = {    self.pos_head:  	np.array([t[0]]).reshape(-1,1), 
													   self.rel:       	np.array([t[1]]).reshape(-1,1), 
													   self.pos_tail:	np.array([t[2]]).reshape(-1,1),
													   self.start_year :np.array([start_lbl]*self.max_rel),
													   self.end_year : np.array([end_lbl]*self.max_rel),
													   self.mode: 			   -1, 
													   self.pred_mode: -1,
													   self.query_mode: -1})
		pos_rel = np.squeeze(pos_rel)

		return pos_head, pos_tail, pos_rel

	def add_placeholders(self):
		self.start_year = tf.placeholder(tf.int32, shape=[None], name = 'start_time')
		self.end_year   = tf.placeholder(tf.int32, shape=[None],name = 'end_time')
		self.pos_head 	= tf.placeholder(tf.int32, [None,1])
		self.pos_tail 	= tf.placeholder(tf.int32, [None,1])
		self.rel      	= tf.placeholder(tf.int32, [None,1])
		self.neg_head 	= tf.placeholder(tf.int32, [None,1])
		self.neg_tail 	= tf.placeholder(tf.int32, [None,1])
		self.mode 	  	= tf.placeholder(tf.int32, shape = ())
		self.pred_mode 	= tf.placeholder(tf.int32, shape = ())
		self.query_mode = tf.placeholder(tf.int32, shape = ())

	def create_feed_dict(self, batch, wLabels=True,dtype='train'):
		ph, pt, r, nh, nt, start_idx = zip(*batch)
		feed_dict = {}
		feed_dict[self.pos_head] = np.array(ph).reshape(-1,1)
		feed_dict[self.pos_tail] = np.array(pt).reshape(-1,1)
		feed_dict[self.rel] = np.array(r).reshape(-1,1)
		feed_dict[self.start_year] = np.array(start_idx)
		# feed_dict[self.end_year]   = np.array(end_idx)
		if dtype == 'train':
			feed_dict[self.neg_head] = np.array(nh).reshape(-1,1)
			feed_dict[self.neg_tail] = np.array(nt).reshape(-1,1)
			feed_dict[self.mode]   	 = 1
			feed_dict[self.pred_mode] = 0
			feed_dict[self.query_mode] = 0
		else: 
			feed_dict[self.mode] = -1

		return feed_dict


	def time_projection(self,data,t):
		inner_prod  = tf.tile(tf.expand_dims(tf.reduce_sum(data*t,axis=1),axis=1),[1,self.p.inp_dim])
		prod 		= t*inner_prod
		data = data - prod
		return data

	def add_model(self):
			#nn_in = self.input_x
		with tf.name_scope("embedding"):
			self.ent_embeddings = tf.get_variable(name = "ent_embedding",  shape = [self.max_ent, self.p.inp_dim], initializer = tf.contrib.layers.xavier_initializer(uniform = False), regularizer=self.regularizer)
			self.rel_embeddings = tf.get_variable(name = "rel_embedding",  shape = [self.max_rel, self.p.inp_dim], initializer = tf.contrib.layers.xavier_initializer(uniform = False), regularizer=self.regularizer)
			self.time_embeddings = tf.get_variable(name = "time_embedding",shape = [self.max_time, self.p.inp_dim], initializer = tf.contrib.layers.xavier_initializer(uniform =False))

		transE_in_dim = self.p.inp_dim
		transE_in     = self.ent_embeddings
		####################------------------------ time aware GCN MODEL ---------------------------##############


	
		## Some transE style model ####
		
		neutral = tf.constant(0)      ## mode = 1 for train mode = -1 test
		test_type = tf.constant(0)    ##  pred_mode = 1 for head -1 for tail
		query_type = tf.constant(0)   ## query mode  =1 for head tail , -1 for rel
		
		def f_train():
			pos_h_e = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.pos_head))
			pos_t_e = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.pos_tail))
			pos_r_e = tf.squeeze(tf.nn.embedding_lookup(self.rel_embeddings, self.rel))
			return pos_h_e, pos_t_e, pos_r_e
		
		def f_test():
			def head_tail_query():
				def f_head():
					e2 = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.pos_tail))
					pos_h_e = transE_in
					pos_t_e = tf.reshape(tf.tile(e2,[self.max_ent]),(self.max_ent, transE_in_dim))
					return pos_h_e, pos_t_e
				
				def f_tail():
					e1 = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.pos_head))
					pos_h_e = tf.reshape(tf.tile(e1,[self.max_ent]),(self.max_ent, transE_in_dim))
					pos_t_e = transE_in
					return pos_h_e, pos_t_e

				pos_h_e, pos_t_e  = tf.cond(self.pred_mode > test_type, f_head, f_tail)
				r  = tf.squeeze(tf.nn.embedding_lookup(self.rel_embeddings,self.rel))
				pos_r_e = tf.reshape(tf.tile(r,[self.max_ent]),(self.max_ent,transE_in_dim))
				return pos_h_e, pos_t_e, pos_r_e
			
			def rel_query():
				e1 = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.pos_head))
				e2 = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.pos_tail))
				pos_h_e = tf.reshape(tf.tile(e1,[self.max_rel]),(self.max_rel, transE_in_dim))
				pos_t_e = tf.reshape(tf.tile(e2,[self.max_rel]),(self.max_rel, transE_in_dim))
				pos_r_e = self.rel_embeddings
				return pos_h_e, pos_t_e, pos_r_e

			pos_h_e, pos_t_e, pos_r_e = tf.cond(self.query_mode > query_type, head_tail_query, rel_query)
			return pos_h_e, pos_t_e, pos_r_e

		pos_h_e, pos_t_e, pos_r_e = tf.cond(self.mode > neutral, f_train, f_test)
		neg_h_e = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.neg_head))
		neg_t_e = tf.squeeze(tf.nn.embedding_lookup(transE_in, self.neg_tail))

		#### ----- time -----###
		t_1 = tf.squeeze(tf.nn.embedding_lookup(self.time_embeddings, self.start_year))
		
		pos_h_e_t_1 = self.time_projection(pos_h_e,t_1)
		neg_h_e_t_1 = self.time_projection(neg_h_e,t_1)
		pos_t_e_t_1 = self.time_projection(pos_t_e,t_1)
		neg_t_e_t_1 = self.time_projection(neg_t_e,t_1)
		pos_r_e_t_1 = self.time_projection(pos_r_e,t_1)
		# pos_r_e_t_1 = pos_r_e

		if self.p.L1_flag:
			pos = tf.reduce_sum(abs(pos_h_e_t_1 + pos_r_e_t_1 - pos_t_e_t_1), 1, keep_dims = True) 
			neg = tf.reduce_sum(abs(neg_h_e_t_1 + pos_r_e_t_1 - neg_t_e_t_1), 1, keep_dims = True) 
			#self.predict = pos
		else:
			pos = tf.reduce_sum((pos_h_e_t_1 + pos_r_e_t_1 - pos_t_e_t_1) ** 2, 1, keep_dims = True) 
			neg = tf.reduce_sum((neg_h_e_t_1 + pos_r_e_t_1 - neg_t_e_t_1) ** 2, 1, keep_dims = True) 
			#self.predict = pos

		'''
		debug_nn([self.pred_mode,self.mode], feed_dict = self.create_feed_dict(self.data[0:self.p.batch_size],dtype='test'))
		'''
		return pos, neg

	def add_loss(self, pos, neg):
		with tf.name_scope('Loss_op'):
			loss     = tf.reduce_sum(tf.maximum(pos - neg + self.p.margin, 0))
			if self.regularizer != None: loss += tf.contrib.layers.apply_regularization(self.regularizer, tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
			return loss

	def add_optimizer(self, loss):
		with tf.name_scope('Optimizer'):
			optimizer = tf.train.AdamOptimizer(self.p.lr)
			train_op  = optimizer.minimize(loss)
		time_normalizer = tf.assign(self.time_embeddings, tf.nn.l2_normalize(self.time_embeddings,dim = 1))
		return train_op

	def __init__(self, params):
		self.p  = params
		self.p.batch_size = self.p.batch_size
		if self.p.l2 == 0.0: 	self.regularizer = None
		else: 			self.regularizer = tf.contrib.layers.l2_regularizer(scale=self.p.l2)
		self.load_data()
		self.nbatches = len(self.data) // self.p.batch_size
		self.add_placeholders()
		self.pos, neg   = self.add_model()
		self.loss      	= self.add_loss(self.pos, neg)
		self.train_op  	= self.add_optimizer(self.loss)
		self.merged_summ = tf.summary.merge_all()
		self.summ_writer = None
		print('model done')

	def run_epoch(self, sess,data,epoch):
		drop_rate = self.p.dropout

		losses = []
		# total_correct, total_cnt = 0, 0

		for step, batch in enumerate(self.getBatches(data, shuffle)):
			feed = self.create_feed_dict(batch)
			l, a = sess.run([self.loss, self.train_op],feed_dict = feed)
			losses.append(l)
		return np.mean(losses)


	def fit(self, sess):
		saver = tf.train.Saver(max_to_keep=None)
		save_dir = 'checkpoints/' + self.p.name + '/'
		if not os.path.exists(save_dir): os.makedirs(save_dir)
		save_dir_results = './results/'+ self.p.name + '/'
		if not os.path.exists(save_dir_results): os.makedirs(save_dir_results)
		if self.p.restore:
			save_path = os.path.join(save_dir, 'epoch_{}'.format(self.p.restore_epoch))
			saver.restore(sess, save_path)
		
		if not self.p.onlyTest:
			print('start fitting')
			validation_data = self.read_valid(self.p.valid_data)
			for epoch in range(self.p.max_epochs):
				l = self.run_epoch(sess,self.data,epoch)
				if epoch%50 == 0:
					print('Epoch {}\tLoss {}\t model {}'.format(epoch,l,self.p.name))
				
				if epoch % self.p.test_freq == 0 and epoch != 0:
					save_path = os.path.join(save_dir, 'epoch_{}'.format(epoch))   ## -- check pointing -- ##
					saver.save(sess=sess, save_path=save_path)
					if epoch == self.p.test_freq:
						f_valid  = open(save_dir_results  +'/valid.txt','w')
					
					fileout_head = open(save_dir_results +'/valid_head_pred_{}.txt'.format(epoch),'w')
					fileout_tail = open(save_dir_results +'/valid_tail_pred_{}.txt'.format(epoch),'w')
					fileout_rel  = open(save_dir_results +'/valid_rel_pred_{}.txt'.format(epoch), 'w')
					for i,t in enumerate(validation_data):
						score = self.calculated_score_for_positive_elements(t, epoch, f_valid, 'valid')
						if score:
							fileout_head.write(' '.join([str(x) for x in score[0]]) + '\n')
							fileout_tail.write(' '.join([str(x) for x in score[1]]) + '\n')
							fileout_rel.write (' '.join([str(x) for x in score[2]] ) + '\n')
				
						if i%500 == 0:
							print('{}. no of valid_triples complete'.format(i))

					fileout_head.close()
					fileout_tail.close()
					fileout_rel.close()
					if epoch ==self.p.test_freq:
						f_valid.close()
					print("Validation Ended")
		else:
			print('start Testing')
			test_data = self.read_valid(self.p.test_data)
			f_test  = open(save_dir_results  +'/test.txt','w')
			fileout_head = open(save_dir_results +'/test_head_pred_{}.txt'.format(self.p.restore_epoch),'w')
			fileout_tail = open(save_dir_results +'/test_tail_pred_{}.txt'.format(self.p.restore_epoch),'w')
			fileout_rel  = open(save_dir_results +'/test_rel_pred_{}.txt'.format(self.p.restore_epoch), 'w')
			for i,t in enumerate(test_data):
				score = self.calculated_score_for_positive_elements(t, self.p.restore_epoch, f_test, 'test')
				fileout_head.write(' '.join([str(x) for x in score[0]]) + '\n')
				fileout_tail.write(' '.join([str(x) for x in score[1]]) + '\n')
				fileout_rel.write (' '.join([str(x) for x in score[2]] ) + '\n')
		
				if i%500 == 0:
					print('{}. no of test_triples complete'.format(i))
			fileout_head.close()
			fileout_tail.close()
			fileout_rel.close()
			print("Test ended")

if __name__== "__main__":
	print('here in main')
	parser = argparse.ArgumentParser(description='HyTE')

	parser.add_argument('-data_type', dest= "data_type", default ='yago', choices = ['yago','wiki_data'], help ='dataset to choose')
	parser.add_argument('-version',dest = 'version', default = 'large', choices = ['large','small'], help = 'data version to choose')
	parser.add_argument('-test_freq', 	 dest="test_freq", 	default = 25,   	type=int, 	help='Batch size')
	parser.add_argument('-neg_sample', 	 dest="M", 		default = 5,   	type=int, 	help='Batch size')
	parser.add_argument('-gpu', 	 dest="gpu", 		default='1',			help='GPU to use')
	parser.add_argument('-name', 	 dest="name", 		default='test_'+str(uuid.uuid4()),help='Name of the run')
	parser.add_argument('-drop',	 dest="dropout", 	default=1.0,  	type=float,	help='Dropout for full connected layer')
	parser.add_argument('-rdrop',	 dest="rec_dropout", 	default=1.0,  	type=float,	help='Recurrent dropout for LSTM')
	parser.add_argument('-lr',	 dest="lr", 		default=0.0001,  type=float,	help='Learning rate')
	parser.add_argument('-lam_1',	 dest="lambda_1", 		default=0.5,  type=float,	help='transE weight')
	parser.add_argument('-lam_2',	 dest="lambda_2", 		default=0.25,  type=float,	help='entitty loss weight')
	parser.add_argument('-margin', 	 dest="margin", 	default=1,   	type=float, 	help='margin')
	parser.add_argument('-batch', 	 dest="batch_size", 	default= 50000,   	type=int, 	help='Batch size')
	parser.add_argument('-epoch', 	 dest="max_epochs", 	default= 5000,   	type=int, 	help='Max epochs')
	parser.add_argument('-l2', 	 dest="l2", 		default=0.0, 	type=float, 	help='L2 regularization')
	parser.add_argument('-seed', 	 dest="seed", 		default=1234, 	type=int, 	help='Seed for randomization')
	parser.add_argument('-inp_dim',  dest="inp_dim", 	default = 128,   	type=int, 	help='Hidden state dimension of Bi-LSTM')
	parser.add_argument('-L1_flag',  dest="L1_flag", 	action='store_false',   	 	help='Hidden state dimension of FC layer')
	parser.add_argument('-onlytransE', dest="onlytransE", 	action='store_true', 		help='Evaluate model on only transE loss')
	parser.add_argument('-onlyTest', dest="onlyTest", 	action='store_true', 		help='Evaluate model for test data')
	parser.add_argument('-restore',	 dest="restore", 	action='store_true', 		help='Restore from the previous best saved model')
	parser.add_argument('-res_epoch',	     dest="restore_epoch", 	default=200,   type =int,		help='Restore from the previous best saved model')
	args = parser.parse_args()
	args.dataset = 'data/'+ args.data_type +'/'+ args.version+'/train.txt'
	args.entity2id = 'data/'+ args.data_type +'/'+ args.version+'/entity2id.txt'
	args.relation2id = 'data/'+ args.data_type +'/'+ args.version+'/relation2id.txt'
	args.valid_data  =  'data/'+ args.data_type +'/'+ args.version+'/valid.txt'
	args.test_data  =  'data/'+ args.data_type +'/'+ args.version+'/test.txt'
	args.triple2id  =   'data/'+ args.data_type +'/'+ args.version+'/triple2id.txt'
	# if not args.restore: args.name = args.name + '_' + time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")
	tf.set_random_seed(args.seed)
	random.seed(args.seed)
	np.random.seed(args.seed)
	set_gpu(args.gpu)
	model  = HyTE(args)
	print('model object created')
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		print('enter fitting')
		model.fit(sess)
