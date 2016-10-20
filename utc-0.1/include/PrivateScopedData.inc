
namespace iUtc{

template<typename T>
PrivateScopedData<T>::PrivateScopedData()
:PrivateScopedDataBase(){
	m_size = 1;
	m_typesize = sizeof(T);
	m_numThreads =0;
	m_userTaskObj = nullptr;
}

template <typename T>
PrivateScopedData<T>::PrivateScopedData(long size)
:PrivateScopedDataBase(){
	m_size = size;
	m_typesize = sizeof(T);
	m_numThreads =0;
	m_userTaskObj = nullptr;
}

template <typename T>
PrivateScopedData<T>::PrivateScopedData(UserTaskBase* userTaskObj)
:PrivateScopedDataBase(){
	m_size = 1;
	m_typesize = sizeof(T);
	m_numThreads =0;
	m_userTaskObj = userTaskObj;
	userTaskObj->registerPrivateScopedData(this);
}

template <typename T>
PrivateScopedData<T>::PrivateScopedData(UserTaskBase* userTaskObj, long size)
:PrivateScopedDataBase(){
	m_size = size;
	m_typesize = sizeof(T);
	m_numThreads = 0;
	m_userTaskObj = userTaskObj;
	userTaskObj->registerPrivateScopedData(this);
}

template <typename T>
PrivateScopedData<T>::~PrivateScopedData(){
	if(m_numThreads !=0){
		std::cerr<<"Error, PrivateScopedData not call destroy!!!\n";
		exit(1);
	}

}

template<typename T>
void PrivateScopedData<T>::init(){
	T* data;
	m_numThreads.fetch_add(1);
	if(!m_dataPtr.get()){
		data = (T*)malloc(m_typesize * m_size);
		m_dataPtr.reset(data);
	}
}

template<typename T>
void PrivateScopedData<T>::destroy(){
	m_numThreads.fetch_sub(1);
	T* data = m_dataPtr.get();
	if(data){
		 free(data);
		 m_dataPtr.release();
	}
	else{
		std::cerr<<"Error, PrivateScopedData call destroy without init!!!\n";
		exit(1);
	}

}

template<typename T>
T* PrivateScopedData<T>::getPtr(){
	return m_dataPtr.get();
}

template<typename T>
int PrivateScopedData<T>::getSize(){
	return m_size;
}

template<typename T>
T PrivateScopedData<T>::load(int index){
	T* data = m_dataPtr.get();
	return data[index];
}

template<typename T>
void PrivateScopedData<T>::store(T value, int index){
	T* data = m_dataPtr.get();
	data[index]= value;
}

template<typename T>
int PrivateScopedData<T>::loadblock(T* dst, int startIdx, int blocks){
	T* data = m_dataPtr.get();
	memcpy(dst, &data[startIdx], blocks*m_typesize);
	return 0;
}

template<typename T>
int PrivateScopedData<T>::storeblock(T* src, int startIdx, int blocks){
	T* data = m_dataPtr.get();
	memcpy(&data[startIdx], src, blocks*m_typesize);
	return 0;
}


template<typename T>
PrivateScopedData<T>::operator T() const{
	T* data = m_dataPtr.get();
	return data[0];
}

template<typename T>
T& PrivateScopedData<T>::operator= (T value){
	T* data = m_dataPtr.get();
	data[0]= value;
	return data[0];
}



}// end namespace iUtc



