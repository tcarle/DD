#include <stdio.h>

typedef struct NET_T {
  int nb_layers;
  int* size0;//size of output
  int* size1;//size of input
  double** layers;
  double** biases;
} net_t;

typedef struct IMG_T{
  int l;
  int ll;
  double* pixels;
} img_t;

net_t load_coeffs(char* file_address){
  FILE* f = fopen(file_address, "r");
  //printf("Opened\n");
  
  net_t network;
  int current_layer;
  int size0, size1;

  char garbage;
  fscanf(f, "%c:", &garbage);
  fscanf(f, "%d", &(network.nb_layers));

  double** n = (double**) malloc(network.nb_layers*sizeof(double*));
  int* s0 = (int*) malloc(network.nb_layers*sizeof(int));
  int* s1 = (int*) malloc(network.nb_layers*sizeof(int));
  double** b = (double**) malloc(network.nb_layers*sizeof(double*));
  network.layers = n;
  network.biases = b;
  network.size0 = s0;
  network.size1 = s1;
  //printf("Entering loop, nb_layers:%d\n", network.nb_layers);
  int old_layer =-1;
  while(!feof(f)){
    fscanf(f, " %c", &garbage); //"L"
    fscanf(f, " %d", &current_layer);
    if(current_layer == old_layer)
      break;
    old_layer=current_layer;
    fscanf(f, " %c", &garbage); // ":"
    fscanf(f, " %d", &size0);
    s0[current_layer] = size0;
    fscanf(f, " %d", &size1);
    s1[current_layer] = size1;
    //printf("Current Layer %d, size0 = %d, size1 = %d\n", current_layer, size0, size1);
    double* tab = (double*) malloc(size0 * size1 * sizeof(double));
    network.layers[current_layer] = tab;
    for(int i = 0; i < size0*size1; i++){
      fscanf(f, "%lf", &(tab[i]));
      //printf("%lf",tab[i]);
    }

    fscanf(f, " %c", &garbage); //"L"
    fscanf(f, " %d", &current_layer);
    fscanf(f, " %c", &garbage); // ":"
    fscanf(f, " %d", &size0);
    //printf("Current Bias %d, size0 = %d\n", current_layer, size0);
    double* tab2 = (double*) malloc(size0 * sizeof(double));
    network.biases[current_layer] = tab2;
    for(int i = 0; i < size0; i++){
      fscanf(f, " %lf", &(tab2[i]));
    }

  }
  fclose(f);
  return network;
}


img_t* load_image(FILE* f){
  img_t* image = (img_t*)malloc(sizeof(img_t));
  image->l = 28;
  image->ll = 28;
  double* img = (double*) malloc(image->l*image->ll*sizeof(double));
  image->pixels = img;
  char garbage;
  fscanf(f, " %c", &garbage);
  for(int i=0; i<image->l*image->ll; i++){
    fscanf(f, " %lf", &(img[i]));
  }
  return image;
}




double relu(double in){
  if(in > 0)
    return in;
  return 0;
}

double* forward(net_t net, img_t img){
  double* in, *out;
  in = (double*)malloc(net.size1[0]*sizeof(double));
  for(int i = 0; i<net.size1[0]; i++){
    in[i]= img.pixels[i];
  //printf("in[%d]=%lf\n", i, in[i]);
  }
  //for(int i=0;i<28;i++)
  //  for(int j =0; j<28; j++)
  //    in[i*8+j] = img.pixels[j*8+i];
  
  for(int l = 0; l <net.nb_layers; l++){
    int size_o = net.size0[l];
    int size_i = net.size1[l];
    out = (double*) malloc(size_o*sizeof(double));
    for(int ligne = 0; ligne < size_o; ligne++){
      out[ligne] = 0;
      for(int col = 0; col < size_i; col++){
	//printf("net.layers[l][ligne+col*]=%lf\n",net.layers[l][ligne+col*size_o]);
	out[ligne]+=in[col]*net.layers[l][ligne*size_i+col];
      }
      out[ligne] = relu(out[ligne]+net.biases[l][ligne]);
    }
    free(in);
    in = out;
   
  }
  return in;
}


void print_image(img_t* im){
  for(int i = 0; i<28*28;i++){
    printf( "%lf ",im->pixels[i]);
  }
  printf("\n\n");
}


int main(){
  char address[256] = "test.txt";
  net_t test = load_coeffs(address);
  char garbage;
  int res[64];

  printf("Reseau ok\n");
  
  FILE* f = fopen("images3.txt", "r");
  img_t* images[64];

  for(int i=0; i<64; i++){
    images[i]=load_image(f);
    //print_image(images[i]);
  }
  printf("Images chargees\n");
  fscanf(f, " %c", &garbage);
  for(int i=0; i<64; i++){
    fscanf(f, "%d", &res[i]);
  }
  
  double* result ;
  int max =-1,i_max=-1;
  int found = 0;
  for(int k =0; k<64; k++){
    result = forward(test, *images[k]);
    max=-1;
    i_max=-1;
    for(int i =0; i < 10; i++){
      printf("proba %d: %lf\n", i, result[i]);
      if(result[i]>max){
	i_max=i;
	max=result[i];
      }
    }
    if(i_max == res[k]){
      found+=1;
    }
  }

  printf("Percentage = %lf\n", (float)found/64);
  // printf("Expected: %d\n", res[0]);
  free(result);
  
  
  return 0;
}
