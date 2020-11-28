#include <stdio.h>
#include <stdlib.h>

int main(){
	int n_GB=0;
	char k;
	int val = 32768;
	unsigned long n_blocks=0;
	unsigned long i;
	int *mem_ptr;
	//unsigned long mem_size=0;
	float mem_size=0;

	printf("This program holds ram memory until 'q' key is pressed.\n");
	printf("How many GigaBytes do you want to hold?\n");
	scanf("%d", &n_GB);

	n_blocks=(unsigned long)(1E9*n_GB)/sizeof(val);
	printf("Allocating %ld blocks of size %ld Bytes...\n\n", n_blocks, sizeof(val));

	
	mem_ptr = (int*) calloc(n_blocks, sizeof(val));
	//mem_ptr = (int*) malloc(n_blocks*sizeof(int));
	mem_size = (n_blocks * sizeof(val))/1E9;

	if (mem_ptr == NULL){
		printf("Memory not allocated.\n");
		exit(0);
	}

	for(i=0; i<n_blocks; i++){
		mem_ptr[i] = 32767;
	}
		
	printf("%0.2f GB holded until 'q' is pressed...\n\n", mem_size);

	while(1){
		scanf("%c", &k);
		if(k == 'q')
		{
			printf("%c pressed. Finishing the program...\n", k);
			break;
		}
	}

	free(mem_ptr);

	return 0;
}
