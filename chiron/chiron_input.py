#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 14:04:57 2017

@author: haotian.teng
"""
import numpy as np
import os,collections,sys
import h5py
import tempfile
raw_labels = collections.namedtuple('raw_labels',['start','end','base'])

# np.random.seed(42)

class Flags(object):
    def __init__(self):
        self.max_reads_number = None
        self.MAXLEN = 1e6 #Maximum Length of the holder in biglist. 1e6 by default
#        self.max_segment_len = 200
FLAGS = Flags()

class biglist(object):
    #Read into memory if reads number < MAXLEN, otherwise read into h5py database
    def __init__(self,data_handle,dtype = 'float32',length = 0,cache = False):
        self.handle = data_handle
        self.dtype = dtype
        self.holder = list()
        self.len = length
        self.cache = cache #Mark if the list has been saved into hdf5 or not
    @property
    def shape(self):
        return self.handle.shape
    def append(self,item):
        self.holder.append(item)
        self.check_save
    def __add__(self,add_list):
        self.holder += add_list
        self.check_save()
        return self
    def __len__(self):
        return self.len + len(self.holder)
    def resize(self,size,axis = 0):
        self.save_rest()
        if self.cache:
            self.handle.resize(size,axis = axis  )
            self.len = len(self.handle)
        else:
            self.holder = self.holder[:size]
    def save_rest(self):
        if self.cache:
            if len(self.holder)!=0:
                self.save()
    def check_save(self):
        if len(self.holder) > FLAGS.MAXLEN:
            self.save()
            self.cache = True

    def save(self):
        if type(self.holder[0]) is list:
            max_sub_len = max([len(sub_a) for sub_a in self.holder])
            shape = self.handle.shape
            for item in self.holder:
                item.extend([0]*(max(shape[1],max_sub_len) - len(item)))
            if max_sub_len > shape[1]:
                self.handle.resize(max_sub_len,axis = 1)
            self.handle.resize(self.len+len(self.holder),axis = 0)
            self.handle[self.len:] = self.holder
            self.len+=len(self.holder)
            del self.holder[:]
            self.holder = list()
        else:
            self.handle.resize(self.len+len(self.holder),axis = 0)
            self.handle[self.len:] = self.holder
            self.len+=len(self.holder)
            del self.holder[:]
            self.holder = list()

    def __getitem__(self,val):
        if self.cache:
            if len(self.holder)!=0:
                self.save()
            return self.handle[val]
        else:
            return self.holder[val]

class DataSet(object):
    def __init__(self,
                 event,
                 event_length,
                 label,
                 label_length,
                 for_eval = False,
                 ):
        """Custruct a DataSet."""
        if for_eval ==False:
            assert len(event)==len(label) and len(event_length)==len(label_length) and len(event)==len(event_length),"Sequence length for event \
            and label does not of event and label should be same, \
            event:%d , label:%d"%(len(event),len(label))
        self._event = event
        self._event_length = event_length
        self._label = label
        self._label_length=label_length
        self._reads_n = len(event)
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._for_eval = for_eval
        self._perm = np.arange(self._reads_n)

    @property
    def event(self):
        return self._event
    @property
    def label(self):
        return self._label
    @property
    def event_length(self):
        return self._event_length
    @property
    def label_length(self):
        return self._label_length
    @property
    def reads_n(self):
        return self._reads_n
    @property
    def index_in_epoch(self):
        return self._index_in_epoch
    @property
    def epochs_completed(self):
        return self._epochs_completed
    @property
    def for_eval(self):
        return self._for_eval
    @property
    def perm(self):
        return self._perm

    def read_into_memory(self, index):
        event = np.asarray(zip([self._event[i] for i in index],[self._event_length[i] for i in index]))
        if not self.for_eval:
            label = np.asarray(zip([self._label[i] for i in index],[self._label_length[i] for i in index]))
        else:
            label = []
        return event,label

    def next_batch(self, batch_size,shuffle = True):
        """Return next batch in batch_size from the data set.
            Input Args:
                batch_size:batch size
                shuffle: boolean, indicate suffle or not
            Output Args:
                inputX,sequence_length,label_batch: tuple of (indx,vals,shape)"""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0:
         if shuffle:
          np.random.shuffle(self._perm)
        # Go to the next epoch
        if start + batch_size > self.reads_n:
          # Finished epoch
          self._epochs_completed += 1
          # Get the rest samples in this epoch
          rest_reads_n = self.reads_n - start
          event_rest_part,label_rest_part = self.read_into_memory(self._perm[start:self._reads_n])

          # Shuffle the data
          if shuffle:
            np.random.shuffle(self._perm)
          # Start next epoch
          start = 0
          self._index_in_epoch = batch_size - rest_reads_n
          end = self._index_in_epoch
          event_new_part,label_new_part = self.read_into_memory(self._perm[start:end])

          if event_rest_part.size > 0:
            event_batch = np.concatenate((event_rest_part, event_new_part), axis=0)
          else:
            event_batch = event_new_part
          if self.for_eval == False and label_rest_part.size > 0:
            label_batch = np.concatenate((label_rest_part, label_new_part), axis=0)
          else:
            label_batch = label_new_part

        else:
          self._index_in_epoch += batch_size
          end = self._index_in_epoch
          event_batch,label_batch = self.read_into_memory(self._perm[start:end])

        if not self._for_eval:
            label_batch = batch2sparse(label_batch)
        seq_length = event_batch[:,1].astype(np.int32)
        return np.vstack(event_batch[:,0]).astype(np.float32),seq_length,label_batch

def read_data_for_eval(file_path,start_index,seg_length,step,smooth_window,skip_step):
    '''
    Input Args:
        file_path: file path to a signal file.
        start_index: the index of the signal start to read.
    '''
    event = []
    event_len = []

    f_signal = read_signal(file_path, smooth_window, skip_step, normalize = True)

    f_signal = f_signal[start_index/skip_step:]
    sig_len = len(f_signal)

    if sig_len > seg_length:
        for indx in range(0, sig_len - seg_length +1, step):
            segment_sig = f_signal[indx:indx+seg_length]
            event.append(segment_sig)
            event_len.append(len(segment_sig))

    evaluation = DataSet(event = event,event_length = event_len,label = [],label_length = [],for_eval = True)
    return evaluation

#def read_cache_dataset(h5py_file_path):
#    """Notice: Return a data reader for a h5py_file, call this function multiple
#    time for parallel reading, this will give you N dependent dataset reader,
#    each reader read independently from the h5py file."""
#    hdf5_record = h5py.File(h5py_file_path,"r")
#    event_h = hdf5_record['event/record']
#    event_length_h = hdf5_record['event/length']
#    label_h = hdf5_record['label/record']
#    label_length_h = hdf5_record['label/length']
#    event_len = len(event_h)
#    label_len = len(label_h)
#    assert len(event_h) == len(event_length_h)
#    assert len(label_h) == len(label_length_h)
#    event = biglist(data_handle = event_h,length = event_len,cache = True)
#    event_length = biglist(data_handle = event_length_h,length = event_len,cache = True)
#    label = biglist(data_handle = label_h,length = label_len,cache = True)
#    label_length = biglist(data_handle = label_length_h,length = label_len,cache = True)
#    return DataSet(event = event,event_length = event_length,label = label,label_length = label_length)

def read_raw_data_sets(data_dir,hdf5_record,seq_length,k_mer,alphabet,jump,smooth_window,skip_step,max_reads_num = FLAGS.max_reads_number):
    ###Read from raw data
#    with h5py.File(h5py_file_path,"a") as hdf5_record :
    event_h = hdf5_record.create_dataset('event/record',dtype = 'float32', shape=(0,seq_length),maxshape = (None,seq_length))
    event_length_h = hdf5_record.create_dataset('event/length',dtype = 'int32',shape=(0,),maxshape =(None,),chunks =True )
    label_h = hdf5_record.create_dataset('label/record',dtype = 'int32',shape = (0,0),maxshape = (None,seq_length))
    label_length_h = hdf5_record.create_dataset('label/length',dtype = 'int32',shape = (0,),maxshape = (None,))
    event = biglist(data_handle = event_h)
    event_length = biglist(data_handle = event_length_h)
    label = biglist(data_handle = label_h)
    label_length = biglist(data_handle = label_length_h)
    count = 0
    file_count = 0
    for name in os.listdir(data_dir):
        if name.endswith(".signal"):
            file_pre = os.path.splitext(name)[0]
            f_signal = read_signal(data_dir+name, smooth_window, skip_step)
            if len(f_signal)==0:
                continue
            try:
                f_label = read_label(data_dir+file_pre+'.label', 0, alphabet, skip_step)
            except:
                sys.stdout.write("Read the label %s fail.Skipped."%(name))
                continue

#            if seq_length<max(f_label.length):
#                print("Sequence length %d is samller than the max raw segment length %d, give a bigger seq_length"\
#                                 %(seq_length,max(f_label.length)))
#                l_indx = range(len(f_label.length))
#                for_sort = zip(l_indx,f_label.length)
#                sorted_array = sorted(for_sort,key = lambda x : x[1],reverse = True)
#                index = sorted_array[0][0]
#                plt.plot(f_signal[f_label.start[index]-100:f_label.start[index]+f_label.length[index]+100])
#                continueholder_
            if jump:
                tmp_event,tmp_event_length,tmp_label,tmp_label_length = read_raw_sliding(f_signal,f_label,seq_length,jump)
            else:
                tmp_event,tmp_event_length,tmp_label,tmp_label_length = read_raw(f_signal,f_label,seq_length)

            event+=tmp_event
            event_length+=tmp_event_length
            label+=tmp_label
            label_length+=tmp_label_length
            del tmp_event
            del tmp_event_length
            del tmp_label
            del tmp_label_length
            count = len(event)
            if file_count%10 ==0:
                if FLAGS.max_reads_number is not None:
                    sys.stdout.write("%d/%d events read.   \n" % (count,FLAGS.max_reads_number))
                    if len(event)>FLAGS.max_reads_number:
                        event.resize(FLAGS.max_reads_number)
                        label.resize(FLAGS.max_reads_number)
                        event_length.resize(FLAGS.max_reads_number)
                        label_length.resize(FLAGS.max_reads_number)
                        break
                else:
                    sys.stdout.write("%d lines read.   \n" % (count))
            file_count+=1
#            print("Successfully read %d"%(file_count))
    assert len(event) == len(event_length)
    assert len(label) == len(label_length)

    train = DataSet(event = event,event_length = event_length,label = label,label_length = label_length)
#    train = read_cache_dataset(h5py_file_path)
    return train

def signal_smoothing(signal, window, step):
    len_signal = len(signal)
    len_smoothed_signal = (len_signal - window) / step +1
    smoothed_signal = np.empty(dtype=np.int32, shape=len_smoothed_signal)

    if window:
        for i,j in enumerate(np.arange(0, len_signal - window +1, step)):
            smoothed_signal[i] = np.median(signal[j : j + window])

    else:
        for i,j in enumerate(np.arange(0, len_signal, step)):
            smoothed_signal[i] = signal[j]

    return smoothed_signal

def read_signal(file_path, smooth_window, skip_step, normalize = True):
    signal = []

    with open(file_path,'r') as f_h:
        for line in f_h:
            for x in line.split():
                signal.append(np.float(x))
        signal = np.asarray(signal)

    len_signal = len(signal)
    if len_signal == 0:
        return signal.tolist()

    if smooth_window or skip_step != 1:
        if len_signal <= smooth_window:
            return [[]]
        signal = signal_smoothing(signal, smooth_window, skip_step)

    if normalize:
        signal = (signal - signal.mean()) / signal.std()

    return signal.tolist()

def base2ind(base, alphabet):
    if len(base) == 5:
        base = base[0]
    return alphabet.index(base.upper())

def read_label(file_path, skip_start, alphabet, skip_step):
    start = []
    end = []
    base = []
    count = 0

    with open(file_path,'r') as f_h:
        for count, line in enumerate(f_h):
            if count < skip_start:
                continue

            record = line.split()
            start.append(int(record[0]) / skip_step)
            end.append(int(record[1]) / skip_step)
            base.append(base2ind(record[2], alphabet))

    return raw_labels(start=start,end=end,base=base)

def read_raw_sliding(raw_signal,raw_label,max_seq_length,jump):
    label_val = []
    label_length = []
    event_val = []
    event_length = []

    len_signal = len(raw_signal)
    first_label_signal = raw_label.start[0]
    start_label = 0
    end = 0

    if len_signal > max_seq_length:
        for indx in range(first_label_signal, len_signal, jump):
            if indx + max_seq_length > len_signal:
                indx = len_signal - max_seq_length
                end = 1

            segment_event = raw_signal[indx : indx + max_seq_length]
            segment_label = []
            tmp_start_label = start_label

            for indw in range(start_label, len(raw_label.end)):
                if raw_label.start[indw] + max_seq_length < raw_label.end[indw]:
                    break

                if raw_label.start[indw] >= indx and raw_label.start[indw] < indx + max_seq_length:
                    segment_label.append(raw_label.base[indw])

                    if raw_label.start[indw] < raw_label.start[start_label] + jump:
                        tmp_start_label = indw

                    if raw_label.end[indw] >= indx + max_seq_length:
                        start_label = tmp_start_label
                        break

            len_segment_label = len(segment_label)

            if len_segment_label >= 3:
                event_val.append(segment_event)
                event_length.append(max_seq_length)
                label_val.append(segment_label)
                label_length.append(len_segment_label)

            if end:
                break

    return event_val, event_length, label_val, label_length

def read_raw(raw_signal,raw_label,max_seq_length):
    label_val = list()
    label_length=list()
    event_val = list()
    event_length = list()
    current_length = 0
    current_label = []
    current_event = []
    for indx,current_start in enumerate(raw_label.start):
        current_end = raw_label.end[indx]
        current_base = raw_label.base[indx]
        segment_length = current_end - current_start
        if current_length+segment_length<max_seq_length:
            current_event += raw_signal[current_start:current_end]
            current_label.append(current_base)
            current_length+= segment_length
        else:
            #Save current event and label, conduct a quality controle step of the label.
            if current_length>(max_seq_length/2) and len(current_label)>3:
                padding(current_event,max_seq_length,raw_signal[current_end:current_end+max_seq_length])
                event_val.append(current_event)
                event_length.append(current_length)
                label_val.append(current_label)
                label_length.append(len(current_label))
            #Begin a new event-label
            current_event = raw_signal[current_start:current_end]
            current_length = segment_length
            current_label = [current_base]
    return event_val,event_length,label_val,label_length

def padding(x,L,padding_list = None):
    """Padding the vector x to length L"""
    len_x = len(x)
    assert len_x<=L, "Length of vector x is larger than the padding length"
    zero_n = L-len_x
    if padding_list is None:
        x.extend([0]*zero_n)
    elif len(padding_list)<zero_n:
        x.extend(padding_list+[0]*(zero_n-len(padding_list)))
    else:
        x.extend(padding_list[0:zero_n])
    return None
def batch2sparse(label_batch):
    """Transfer a batch of label to a sparse tensor"""
    values = []
    indices = []
    for batch_i,label_list in enumerate(label_batch[:,0]):
        for indx,label in enumerate(label_list):
            if indx>=label_batch[batch_i,1]:
                break
            indices.append([batch_i,indx])
            values.append(label)
    shape = [len(label_batch),max(label_batch[:,1])]
    return (indices,values,shape)

#
def main():
### Input Test ###
	Data_dir = "/media/haotianteng/Linux_ex/Nanopore_data/Lambda_R9.4/raw/"
	train = read_raw_data_sets(Data_dir,seq_length = 400)
	for i in range(100):
	    inputX,sequence_length,label = train.next_batch(10)
	    indxs,values,shape = label
if __name__=='__main__':
    main()
#
#
#
#hdf5_record = h5py.File('/home/haotianteng/Documents/123/test2.hdf5',"w")
#event_h = hdf5_record.create_dataset('test2',dtype = 'float32', shape=(0,300),maxshape = (None,300))
