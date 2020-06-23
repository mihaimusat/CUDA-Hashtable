#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>

#include "gpu_hashtable.hpp"

__global__ void kernel_reshape(hash_entry *old_list, int old_size, hash_entry *new_list, int new_size) { 
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int old_key;
	int new_key;
	
	int index;
	
	// verific daca idx calculat anterior este valid
	if (idx >= old_size)
		return;

	// daca am gasit un slot cu cheia 0, il ignor
	if (old_list[idx].key == 0)
		return;

	// calculez cheia pe care trebuie sa o inserez
	new_key = old_list[idx].key;

	// caut un slot liber folosind linear probing
	index = hash1(new_key, new_size);

	// mut valorile din vechiul hashtable in noul hashtable in mod atomic
	while (true) {
		
		// incerc sa introduc noua cheie pe pozitia curenta
		old_key = atomicCAS(&(new_list[index].key), KEY_INVALID, new_key);

		// daca am reusit, introduc si valoarea corespunzatoare din vechiul hashtable
		if (old_key == 0) {
			new_list[index].value = old_list[idx].value;
			return;
		}

		// altfel, cresc circular indexul
		else {
			index = (index + 1) % new_size;
		}
	}
}

__global__ void kernel_insert(hash_entry *list, int size, int *keys, int *values, int numKeys) {
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int new_key;
	int old_key;
	
	int index;

	// verific daca idx calculat anterior este valid
	if (idx >= numKeys)
		return;

	// daca cheia sau valoarea sunt negative, le ignor
	if (keys[idx] <= 0 || values[idx] <= 0)
		return;
	
	// calculez noua cheie pe care trebuie sa o inserez 
	new_key = keys[idx];
	
	// caut un slot liber folosind linear probing
	index = hash1(keys[idx], size);

	// inserez perechea (key, value) in lista in mod atomic
	while (true) {
		
		// incerc sa introduc noua cheie pe pozitia curenta
		old_key = atomicCAS(&(list[index].key), KEY_INVALID, new_key);

		// daca am o cheie noua sau e aceeasi cheie, introduc si valoarea 
		if (old_key == 0 || old_key == new_key) {
			list[index].value = values[idx];
			return;
		}
	
		// altfel, cresc circular indexul
		else {
			index = (index + 1) % size;
		}	
	}
}

__global__ void kernel_get(hash_entry *list, int size, int *keys, int *values, int numKeys) {
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	int index;
	int aux;
	
	// verific daca idx calculat anterior este valid
	if (idx >= numKeys)
		return;

	// caut un slot liber folosind linear probing
	index = hash1(keys[idx], size);
	
	// salvez pozitia obtinuta anterior
	aux = index;

	// daca cheia e negativa, o ignor
	if (keys[idx] <= 0) {
		values[idx] = 0;
                return;
	}

	// obtin valoarea corespunzatoare cheii curente in mod atomic 
	while (true) {
	
		// daca am gasit un slot cu cheia 0, am terminat
		if (list[index].key == 0) {
			values[idx] = 0;
			return;
		}
		
		// daca am gasit cheia, salvez valoarea
		else if (list[index].key == keys[idx]) {
			values[idx] = list[index].value;
			return;
		}

		else {
			// altfel, cresc indexul in mod circular
			index = (index + 1) % size;

			// daca indexul curent este egal cu cel initial, 
			// nu am gasit cheia in lista
			if (aux == index) {
				values[idx] = 0;
				return;
			}
		}
	}
}

/* INIT HASH
 */
GpuHashTable::GpuHashTable(int size) {

	cudaMalloc(&hashmap.list, size * sizeof(hash_entry));
	if (hashmap.list == NULL) {
		printf("Error: Failed to allocate hashtable list\n");
		return;
	}

	cudaMemset(hashmap.list, 0, size * sizeof(hash_entry));
	
	hashmap.occupied_slots = 0;
	hashmap.available_slots = size;
}

/* DESTROY HASH
 */
GpuHashTable::~GpuHashTable() {

	cudaFree(hashmap.list);
	hashmap.list = NULL;
	hashmap.available_slots = 0;
	hashmap.occupied_slots = 0;
}

/* RESHAPE HASH
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	
	hash_entry *new_list;
	
	// aloc o lista de perechi de dimensiune numBucketsReshape pentru noul hashtable
	cudaMalloc(&new_list, numBucketsReshape * sizeof(hash_entry));
	if (new_list == NULL) {
		printf("Error: Failed to allocate new hashtable list\n");
		return;
	}

	cudaMemset(new_list, 0, numBucketsReshape * sizeof(hash_entry));

	// daca lista vechiului hashtable e goala, lista noului hashtable e cea alocata anterior
	if (hashmap.occupied_slots == 0) {
		cudaFree(hashmap.list);
		hashmap.list = new_list;
		hashmap.available_slots = numBucketsReshape;
		return;
	}	

	// calculez numarul de blocuri necesare pentru a rula kernelul 
	int num_blocks = hashmap.available_slots / THREADS_PER_BLOCK;
	if (hashmap.available_slots % THREADS_PER_BLOCK != 0) 
		num_blocks++;
	
	// mut valorile din vechiul hashtable in noul hashtable
	kernel_reshape<<<num_blocks, THREADS_PER_BLOCK>>>(hashmap.list, hashmap.available_slots, new_list, numBucketsReshape);

	cudaDeviceSynchronize();

	// eliberez memoria ocupata de lista vechiului hashtable
	cudaFree(hashmap.list);
	hashmap.list = new_list;
	new_list = NULL;

	// actualizez numarul de locuri disponibile in hashtable la numBucketsReshape
	hashmap.available_slots = numBucketsReshape;
}

/* INSERT BATCH
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	
	int *device_keys;
	int *device_values;

	// aloc memorie pentru chei si valori pentru a le transmite kernelului
	cudaMalloc(&device_keys, numKeys * sizeof(int));
	cudaMalloc(&device_values, numKeys * sizeof(int));

	if (device_keys == NULL || device_values == NULL) {
		printf("Error: Failed to allocate device_keys or device_values\n");
		return false;
	}

	// copiez chei si valori din RAM in VRAM
	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	// verific load factor ca sa vad daca am nevoie de un reshape inainte de a le insera
	if ((numKeys + hashmap.occupied_slots) / hashmap.available_slots > MIN_LOAD)
		reshape((int)1.5 * (numKeys + hashmap.occupied_slots) / MIN_LOAD);

	// calculez numarul de blocuri necesare pentru a rula kernelul 
	int num_blocks = numKeys / THREADS_PER_BLOCK;
	if (numKeys % THREADS_PER_BLOCK != 0) 
		num_blocks++;

	// inserez perechea (key, value) in lista
	kernel_insert<<<num_blocks, THREADS_PER_BLOCK>>>(hashmap.list, hashmap.available_slots,
							 device_keys, device_values, numKeys);
	
	cudaDeviceSynchronize();

	// actualizez numarul de locuri ocupate din hashtable
	hashmap.occupied_slots += numKeys;
	
	// eliberez memoria ocupata de chei si valori in VRAM
	cudaFree(device_keys);
	cudaFree(device_values);

	return true;
}

/* GET BATCH
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	
	int *device_keys;
	int *device_values;
	int *host_values;

	// aloc memorie pentru chei si valori pentru a le transmite kernelului
	cudaMalloc(&device_keys, numKeys * sizeof(int));
	cudaMalloc(&device_values, numKeys * sizeof(int));

	if (device_keys == NULL || device_values == NULL) {
		printf("Error: Failed to allocate device_keys or device_values\n");
		return NULL;
	}
	
	// copiez chei din RAM in VRAM
	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);	

	// calculez numarul de blocuri necesare pentru a rula kernelul  
	int num_blocks = numKeys / THREADS_PER_BLOCK;
	if (numKeys % THREADS_PER_BLOCK != 0) 
		num_blocks++;
	
	// obtin valorile corespunzatoare cheilor din hashtable
	kernel_get<<<num_blocks, THREADS_PER_BLOCK>>>(hashmap.list, hashmap.available_slots, device_keys, device_values, numKeys);
	
	cudaDeviceSynchronize();

	// copiez valorile intoarse de kernel din VRAM in RAM
	host_values = (int*)malloc(numKeys * sizeof(int));
	cudaMemcpy(host_values, device_values, sizeof(int) * numKeys, cudaMemcpyDeviceToHost);
	
	// eliberez memoria ocupata de chei si valori in VRAM
	cudaFree(device_keys);
	cudaFree(device_values);

	return host_values;
}

/* GET LOAD FACTOR
 * num elements / hash total slots elements
 */
float GpuHashTable::loadFactor() {
	
	// no larger than 1.0f = 100%	
	if (hashmap.available_slots == 0)
		return 0;
	return (1.0f * hashmap.occupied_slots) / hashmap.available_slots;
}

/*********************************************************/

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
