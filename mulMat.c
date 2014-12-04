//  Created by Alberto Penhos and Eduardo Rubinstein on 20/11/2014.
//  Copyright (c) 2014 Alberto Penhos and Eduardo Rubinstein. All rights reserved.
//

/*
 * Referencias:
 * http://www.cs.hofstra.edu/~cscccl/csc145/imul.c
 * http://www.arc.vt.edu/resources/software/openmp/docs/openmp_mmult.c
 * http://www.eecis.udel.edu/~cavazos/cisc879/Lecture-03.pdf
 * http://biogrid.engr.uconn.edu/REU/Reports_10/Final_Reports/Rifat.pdf
 * http://www.dcc.fc.up.pt/~fds/aulas/PPD/1112/mpi_openmp.pdf
*/

#include <stdio.h>
#include <mpi.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>
#include <math.h>

/*los mensajes que le puede mandar el maestro a los esclavos
es preferible ocupar las palabras reservadas (para saber que hace cada cosa ) y multiplataformas
*/
#define MSG_JOB		0
#define MSG_END		1	
#define MSG_MATINFO	2
#define MSG_DATA	3
#define MSG_FINALIZE	4
#define MSG_HANDSHAKE	5
#define MSG_REDUCE_C	6

/*Matrices a calcular antes de parar el programa*/
#define COMPLETE_LIMIT 20

/*Tamaño de la matriz para evitar la sobrecarga de memoria*/
#define MATRIX_SIZE_LIMIT 10

/*Valores dentro de la matriz menores al definido */
#define MAX_VALUE 10

/*Tiempo de espera en caso de que falle un proceso */
#define TIMEOUT 0.01

/*Constantes para la distribucion de Poisson ... NO REDUCIR BETA YA QUE GENERA FALSOS POSITIVOS DE FALLA*/
#define alpha 30.
#define beta .25
#define gamma 2.5
#define detla 500.
#define c 0.95

/*Communicador nuevo de MPI que representa los procesos no fallidos */
MPI_Comm new_comm;

/*Nodo de lista ligada para la cola*/
typedef struct _matrixNode{
  int m,n;
  int *M;
  struct _matrixNode *next;
} matrixNode;

/*inicializacion de la lista*/
matrixNode *matListStart = NULL;

/*Variables globales para saber
 * 1.) matrices en la cola
 * 2.) matrices recilcadas por falla
 * 3.) matrices multiplicadas*/
int queueSize = 0, recycled = 0, completed = 0;

/*variable para cuando el master quiere hacer restart*/
int RB;

/*El rank inicial por si llega a fallar un proceso ej si el proceso 2 falla 
el proceso 3 se convierte en el 2 en el nuevo grupo*/
int initialRank;

/*Manejador de señal para quitarnos del grupo de los vivos*/
void sig_handler(int signo)
{
  if (signo == SIGINT || signo == SIGTERM){
/*MPI_Split toma un grupo y lo divide en subgrupos de acuerdo al segundo argumento que es pasado, divide el proceso en vivos y muertos*/
    MPI_Comm_split(new_comm, 0, initialRank, &new_comm);//el 0 se queda con los muertos y pasa a los vivos
    MPI_Finalize( );
    exit(0);
  }
  
}

float Poisson_interval( float x ){//funcion para calcular el intervalo de Poisson
  
  float R = -(float)rand( ) / (float)RAND_MAX;
  float X = 1./x*log(1-R);
  return X;
}

void printMatrix( int * M, int d1, int d2){//funcion para imprimir las matrices
  int i;
  
  fprintf(stderr,"==================\n");//stderr es ocupado para imprimir en la computadora maestra
  for(i = 0; i < d1*d2; i++){
    if( i > 0 && i%d2 == 0 )
      fprintf(stderr,"\n");
    fprintf(stderr,"%d ", M[i]);
  }
  fprintf(stderr,"\n==================\n");
  
}

/*funcion que agraga matrices a la cola */
void addToMatList( int *M, int m, int n){
  //fprintf(stderr,"Agregando [%d x %d] matriz a la cola\n", m, n); 
  queueSize++;
  
  /* Ocupamos LIFO */
  matrixNode *tmp = (matrixNode *)malloc(sizeof(matrixNode));
  tmp->M = M;//matrix de mxn
  tmp->m = m;//dimensiones
  tmp->n = n;//dimensiones
  tmp->next = matListStart;
  matListStart = tmp;// apunta al mas nuevo Last In First Out
}/*lista ligada consiste de matrixNode que consiste de informacion de matriz e ints para su tamaño*/


/*Revisa si hay una matriz para multiplicar en la lista*/
matrixNode *pairInList( int rows, int columns){
  /*busca la lista para ver si hay una matriz que pueda ser multiplicada con la matriz llegada*/
  matrixNode *tmp = matListStart;
  
  if( matListStart == NULL)
    return NULL;
  
  /*Va revisando los elementos para encontrar el que se pueda multiplicar */
  if( matListStart->m == columns || matListStart->n == rows){
    //fprintf(stderr,"Quitar [%d x %d] matriz de la cola\n",tmp->m, tmp->n);
    queueSize--;
    /*Quitamos la matriz de la cola y la devolvemos*/
    matListStart = matListStart->next;
    return tmp;
  }
  
  /*Busca otro incidencia */
  while( tmp->next != NULL ){
    if( tmp->next->m == columns || matListStart->next->n == rows){
      /*Necesitamos dos apuntadores para poder quitar la matriz de la cola*/
      matrixNode *tmp2 = tmp->next;
      tmp->next = tmp->next->next;
      fprintf(stderr,"Quita [%d x %d] matriz de la cola\n",tmp2->m, tmp2->n);
      queueSize--;
      /*Regresa el elemento */
      return tmp2;
    }
    tmp = tmp->next;
  }
  /*Si regresa NULL no encontro*/
  return NULL;
}


int * generate_random_matrix( int val1, int val2){
  
  int i, *tmp = (int *)malloc(val1*val2*sizeof(int));
  
  /*Inicializamos en un valor random */
  for( i = 0; i < val1*val2; i++)
    tmp[i] = rand( ) % MAX_VALUE;
  
  return tmp;
}

int * allocate_matrix( int val1, int val2){
  //usamos este para guardar los resultados the la matriz
  int *tmp = (int *)malloc(val1*val2 * sizeof(int));
  int i;
  /*Inicializamos en  0*/
  for(i = 0; i < val1*val2; i++)
    tmp[i] = 0;
  
  if( tmp == NULL){
    fprintf(stderr,"error de Malloc.\n");
    exit(1);
  }
  
  return tmp;
  
}

/* Cumple con la regla de la multiplicacion en caso de que A*B no se pueda y B*A si */
void swapMatrices( int d[4], int **A, int **B){
  
  int tmp1 = d[0], tmp2 = d[3];
  d[0] = d[2];
  d[3] = d[1];
  d[2] = tmp1;
  d[1] = tmp2;
  
  int *tmp3 = *A;
  *A = *B;
  *B = tmp3;
}

/* Manejamos las dos matrices */
int handleMatrices( int **A, int **B, int d[4] ){
  /*mandada a llamar despus de la generacion de dos matrices random y checa si se pueden multiplicar o no*/
  int notFound = 0;
  /*se checa si son compatibles si no se trata de cambiar una si siguen sin ser, se checa si en la lista hay una para la A y luego para B*/
  /*revisa A*B */
  while( d[1] != d[2] && !notFound){//si despues del cambio de matriz se necesita hacer el swap se hace
    /*revisa B*A*/
    if( d[3] == d[0] )
      swapMatrices(d, A, B);
    else{
      /*Si no son validas buscas en la lista para encontrar un match A*/
      matrixNode *tmp1 = pairInList( d[0], d[1] );
      if( tmp1 != NULL ){
	/*Si encuentaras una , pones B en la lista y cambias B por la nueva*/
	addToMatList( *B, d[2], d[3] );
	d[2] = tmp1->m;
	d[3] = tmp1->n;
	*B = tmp1->M;
	free(tmp1);
      }
      else{
	/*Si las A no encuentra parajea busca por  la B*/
	addToMatList( *A, d[0], d[1] );
	tmp1 = pairInList( d[2], d[3] );
	if( tmp1 != NULL ){//NULL si no hay pareja y not NULL si hay 
	  d[0] = tmp1->m;
	  d[1] = tmp1->n;
	  *A = tmp1->M;
	  free(tmp1);
	}
	else{
	  /*Si no se encuentran matrices a multiplicar pones las dos en espera y genera nuevas matrices*/
	  addToMatList( *B, d[2], d[3] );
	  notFound = 1;
	}
      }
    }
  }
  return notFound;
}

/*se hace un timeout para ver si un proceso responde en un tiempo y si no lo consideramos como un 
proceso fallido timeout= 1/beta*/ 
int timeout_recv(int source ){
  
  double t1, t2; 
  /*Empieza a contar el tiempo */
  t1 = MPI_Wtime(); 
  int flag = 0;
  int buf = 0;
  float RC = 0;
  //request y status los ocupamos para probar el MPI_test
  MPI_Request req;
  MPI_Status stat;
  
//irecv no se bloquea como MPI_REcv
  MPI_Irecv( &buf, 1, MPI_INT, source, MSG_HANDSHAKE, new_comm, &req);
  
  /*Checamos el tiempo y si hay respuesta*/
  t2 = MPI_Wtime();
  
  if( (float)rand( )/(float)RAND_MAX < c )
    RC = 1;
  
  /*Espera hasta que se reciba el mesnaje o haya un tiempofuera*/
  while( !flag && t2 - t1 < TIMEOUT ){
    t2 = MPI_Wtime();
    /*revisa que este completo*/
    MPI_Test(&req, &flag, &stat);//confirma que recivimos un mensaje
    /*si el maestro decide  RC,espera un intervalo de  Poisson de variancia b*/
    if( !flag && t2 - t1 >= TIMEOUT && RC ){
      sleep( Poisson_interval(1./beta) );//el 1 es flotante
      
      /*Reinicia el timer*/
      t1 = t2;
      /* RC=recupera*/
      RC = 0;
    }
  }  
  /*MPI_Test( ) regresa positivo si el flag es ==1 y se recupero con exito */
  return flag;
}

/*Se manda un mensaje y se espera una respueta, si no hay respuesta decimos que fallo*/
int shakeHands( int size ){
  //manda saludos a los esclavos y ocupa timeout_recv para decidir si esta vivo o no
  int i, failCount = 0, res, where;
  for(i = 1; i < size; i++){
    /*manda el mensaje del saludo */
    MPI_Send(&i, 1, MPI_INT, i, MSG_HANDSHAKE, new_comm);
    /*vemos si tenemos respuesta*/
    res = timeout_recv(i);
    if( res < 1 ){//si recibe <1 esta muerto de otra manera esta vivo
      where = i;
      /*ver si reseteamos con probabilidad 1/c */
      if( (float)rand( )/(float)RAND_MAX > c )
	RB = 1;
      //todos los procesos tiene 3 data streams stdin, stdout y stderr stdin es entrada keyboard, stdout &stderr son usualmente de terminal, stderr lo va a imprimir en la terminal maestra
      fprintf(stderr,"El maester decidio que el esclavo  %d fallo.\n",i);
      failCount++;
    }
  }
  /*Si mas de un esclavo fallo el maestro mata el programa */
  /*si uno fallo se regresa el rank*/
  /*Si no hay falla regresa 0 */
  if( failCount > 1)
    return -1;
  else if( failCount == 1)
    return where;
  else
    return 0;
}

/*Mensaje a los procesos vivos */
void sendToAll( int *buf, int bsize, int msgType, int size){
  
  int i;
  for(i = 1; i < size; i++)
    MPI_Send(buf, bsize, MPI_INT, i, msgType, new_comm);
}

/*Calcular que esclavo va a hacer que parte del trabajo*/
int calculateJob( int i, int msg[4], int d[4], int *nelems, int currSize){
  /*para la multiplicacion todos los esclavos toman la matriz B y cada uno toma un pedazo de la A*/
  /*Elemento principal del esclavo */
  msg[0] = *nelems*(i - 1);
  
  int start = *nelems;
  
  /*Las tareas se dividen para todos los esclavos y al ultimo esclavo se le sumaran los elementos restantes */
  if( i == currSize - 1)
    *nelems += d[0]*d[1] % (currSize - 1);
  
  /*Numero de elementos asignados al esclavo */
  msg[1] = *nelems;

  return i;
  
}

int check_for_failures( int size, int rank, int *A, int *B, int d[4] ){
  
  /*iniciamos el saludo*/
  int failures = shakeHands( size );
  /*No se permite mas de una falla a la vez porque en caso de mas de una el maestro mata el sistema*/
  if( failures < 0 ){
    fprintf(stderr,"Falla de sistema no se permite mas que una falla a la vez\n");
    MPI_Abort(MPI_COMM_WORLD, 0);
    exit(0);
    return;
  }
  
  /*en caso de reinicio ponemos las matrices en la lista y se recilcan las matrices*/
  if( RB ){
    addToMatList(A, d[0], d[1] );
    addToMatList(B, d[2], d[3] );
    sleep(1./alpha);//1. es porque es numero flotante
    recycled += 2;
  }
  
  return failures;
}

void master_process(int rank, int size){
  /*es la rutina del master*/
  int *A, *B, i, d[4], currSize = size;
  int msg[4];
  
  /* toma un pedazo de la funcion de tiempo aleatroio */
  srand(time(NULL));
  
  while( completed < COMPLETE_LIMIT ){
    
    sleep(2);
    
    /*Genera 4 dimensiones aleatorias y las llena*/
    d[0] = (rand( ) ) % MATRIX_SIZE_LIMIT + size; 
    d[1] = (rand( ) ) % MATRIX_SIZE_LIMIT + size;
    d[2] = (rand( ) ) % MATRIX_SIZE_LIMIT + size;
    d[3] = (rand( ) ) % MATRIX_SIZE_LIMIT + size;
    
    A = generate_random_matrix( d[0],d[1] );//dimenciones de la primera matriz
    B = generate_random_matrix( d[2],d[3] );//dimenciones de B
    
    int notFound = handleMatrices(&A,&B,d);
    
    /*Si no se pueden multiplicar y no encuentra matriz compatible en la lista vuelve a empezar*/
    if( notFound )
      continue;
    
    fprintf(stderr,"Matriz A:\n");
    printMatrix(A, d[0], d[1]);//impresion matriz A
    fprintf(stderr,"Matriz B:\n");
    printMatrix(B, d[2], d[3]);
    
    int *C = allocate_matrix( d[0], d[3] );
    int *C2 = allocate_matrix( d[0], d[3] );
    
    check_for_failures( size, rank, A, B, d );
    /*el master manda a llamar a los vivos*/
    MPI_Comm_split(new_comm, 1, rank, &new_comm);
    /*obtiene el nuevo tamaño del grupo*/
    MPI_Comm_size(new_comm, &size);
    
    if( size == 1 ){
      fprintf(stderr,"Regresa por la falla de proceso.\n");
      return;
    }
    if( RB ){
      RB = 0;
      continue;
    }
    
    /*Manda el tamaño de la matriz a los esclavos*/   
    sendToAll(d, 4, MSG_MATINFO, size);
    /*Manda la matriz B a los esclavos */
    sendToAll(B, d[2]*d[3], 0, size);
    
    /*Divide matriz A en [sizeof A]/#numero de esclavos y le da su porcion a cada uno*/
    int nelems = d[0]*d[1]/(size - 1);//cuantos elementos le toca a cada esclavo
    int i;
    
    for(i = 1; i < size; i++){
      
      calculateJob(i, msg, d, &nelems, size);
      
      /*Informacion del trabajo*/
      MPI_Send(msg, 4, MPI_INT, i, MSG_JOB, new_comm );
      /*Mandar los datos del trabajo*/
      MPI_Send(&A[msg[0]], nelems, MPI_INT, i, MSG_DATA, new_comm);
    }
    
    int failed = check_for_failures( size, rank, A, B, d );
    
    /*Si un esclavo falla le damos su trabajo a otro*/
    if( failed > 0 && size - 1 > 1){
      /*Volvemos a clacular esa parte*/
      calculateJob(failed, msg, d, &nelems, size);
      /*MPI split para genera un grupo de vivos*/
      MPI_Comm_split(new_comm, 1, rank, &new_comm);//1 porque el master no muere y porqeu llama a los vivos
      /*Nuevo tamaño de grupo*/
      MPI_Comm_size(new_comm, &size);
      //decidimos a que proceso se le da su trabajo usualmente el que toma su rank toma su trabajo
	int newTarget = (failed ) % size;
      /*El maestro no hace la tarea la realiza un esclavo*/
      if( newTarget == 0)
	newTarget++;
      /*Manda el mensaje del trabajo*/
      MPI_Send(msg, 4, MPI_INT, newTarget, MSG_JOB, new_comm );
      /*Manda los datos del trabajo*/
      MPI_Send(&A[msg[0]], nelems, MPI_INT, newTarget, MSG_DATA, new_comm);
    }
    else{
      //si no hay falla se manda igual 
	MPI_Comm_split(new_comm, 1, rank, &new_comm);
      MPI_Comm_size(new_comm, &size);//size numero de procesos vivos
      if( size == 1 ){
	fprintf(stderr,"Falla del proceso.\n");
	return;
      }
      if( RB ){
	RB = 0;
	continue;
      }
    }

    
    
    sendToAll(msg, 4, MSG_REDUCE_C, size);
    /*Recolectamos la informacion de los esclavos usando la operacion reduction additive*/
    MPI_Reduce(C, C2, d[0]*d[3], MPI_INT, MPI_SUM, 0, new_comm);
    
    completed++;
    /*Si completamos los mandados salimos*/
    int tag = completed < COMPLETE_LIMIT ? MSG_END : MSG_FINALIZE;
    for(i = 1; i < size; i++)
      MPI_Send(d, 4, MPI_INT, i, tag, new_comm );
    
    fprintf(stderr, "========\nCalculacion de la matriz calculada %d, con dimensiones %d x %d.\n", completed, d[0], d[3] );
    fprintf(stderr, "Matrices en espera: %d\n", queueSize);
    fprintf(stderr, "Matrices recicladas: %d\n", recycled);
    fprintf(stderr, "tamano actual : %d\n", size);
    fprintf(stderr,"Resultado:\n");
    printMatrix(C2, d[0], d[3] );
    free(A); free(B);
    free(C); free(C2);
  }
  
}

void execute_job_multithreaded( int *A, int *B, int *C, int msg[2], int d[4] ){
  /*es la ejecucion de 1 trabajo */
  int nelems = msg[1];
  int start = msg[0];
  int i, j;
  
  /*C es la reduccion todos los threads deben de sumar su reusltados al final*/
  //fistprivate cada thread tiene su propia copia y se inicializa en el valor de antes del pragma, private data es mas rapida que shared data
//cada proceso recibe nelements de la matrz A y por cada uno lo tiene que multiplicar por d3 de B , d3 representa la columna en B
//los ints el / nos da la linea del elemneto y % nos da la columna del elemento
  #pragma omp parallel reduction(C) firstprivate(start,nelems,d,A,B)
  for(i = 0; i < nelems; i++){
    int x = (start + i) / d[1];
    int y = (start + i) % d[1];
    for( j = 0; j < d[3]; j++)
      C[x*d[3] + j] += A[i]*B[y*d[3] + j];
  }
}

/*Significa que el proceso puede fallar usamos una distribucion uniforme 1/#procesos de falla*/
void possible_fail( int size, int rank){//las lineas estan comentadas para poder terminar el programa via terminal
  
  /*int r = rand( )%size;
  if( r == rank ){
    fprintf(stderr,"%d fallo.\n",rank);
    sig_handler( SIGINT);
  }*/
}
void slave_process(int rank, int size){
  //todo esclavo define el handler
/*definimos el handle que interrumpe la señal, y contamos con MPI_Wtime*/
  double t2,t1 = MPI_Wtime( );//espera el mensaje del master
  signal(SIGINT, sig_handler);
  signal(SIGTERM, sig_handler);
  
  while( 1 ){//recibe mensaje y ejecuta de acuerdo a eso
    int * A = NULL, *B = NULL, *C = NULL, d[4];
    int i, j, jobDone = 0;
    
    /*Esperamos a que termine la multipolicacion*/
    while( !jobDone ){
      
      int msg[4];
      MPI_Status stat;
      /*Recibimos mensaje indefinido*/
      MPI_Recv( msg, 4, MPI_INT, 0, MPI_ANY_TAG, new_comm, &stat);//espera un mensaje de 4 iint el MPI tag va del 1-6 y stat es una estructura de tipo Status  
      
      /*Tipo de mensaje*/
      int type = stat.MPI_TAG;//guarda el tag real
      
      switch( type ){
	
	case MSG_MATINFO://mensaje que manda el maestro donde dice la dimensiones de las matrices
	  /*informacion recibida por la multiplicacion de matrices*/
	  d[0] = msg[0]; d[1] = msg[1];
	  d[2] = msg[2]; d[3] = msg[3];
	  /*por si se reinicio el proceso y no quitamos de memoria la matriz*/
	  if( B != NULL ){
	    free(B);
	    B = NULL;
	  }
	  if( C != NULL ){
	    free(C);
	    C = NULL;
	  }
	  B = allocate_matrix( d[2], d[3] );
	  C = allocate_matrix( d[0], d[3] );
	  
	  /*recibimos matriz B*/
	  MPI_Recv( B, d[2]*d[3], MPI_INT, 0, MPI_ANY_TAG, new_comm, &stat);//recive la informacion de la matriz B
	  break;
	  
	case MSG_JOB://el master mando un trabajo
	  //fprintf(stderr,"%d recibo de %d a %d, del total %d\n", rank,msg[0], msg[0]+msg[1], d[0]*d[1]);
	  /*reservamos memoria para propia informaicon*/
	  A = allocate_matrix( msg[1], 1);
	  
	  /*Recive info del trabajo*/
	  MPI_Recv( A, msg[1], MPI_INT, 0, MSG_DATA, new_comm, &stat);//msg1 elementos para calcular
	  
	  execute_job_multithreaded(A, B, C, msg, d);
	  
	  free(A);
	  
	  break;
	  
	case MSG_END:
	  /*Significa que termino el trabajo*/
	  jobDone = 1;
	  break;
	  
	case MSG_FINALIZE://el systema termino
	  return;
	case MSG_HANDSHAKE://recibimos el handshake y se llama a la funcion possible_fail y ver si le responde
	  
	  /*Checa si el intervalo Poisson paso si si revisamos las fallas*/
	  t2 = MPI_Wtime( );
	  if( t2 - t1 > Poisson_interval( gamma ) )
	    possible_fail( size, rank);
	  
	  /*Respondemos el saludo*/
	  MPI_Send(&i, 1, MPI_INT, 0, MSG_HANDSHAKE, new_comm );
	  /*Si llegamos aqui quiere decir que estamos en los procesos vivos*/
	  MPI_Comm_split(new_comm, 1, rank, &new_comm);
	  /*nuestro rank en este grupo*/
	  MPI_Comm_rank( new_comm, &rank);
	  break;
	  
	case MSG_REDUCE_C://mandar el trabajo de vuelta al profesor
	  /*Manda los resultados al maestro*/
	  MPI_Reduce(C, C, d[0]*d[3], MPI_INT, MPI_SUM, 0, new_comm);
	  break;
	  
      }
    }
    free(B);
    free(C);
    
  }
}
int main( int argc, char * argv[] ){
  
  MPI_Init (&argc, &argv); //todos los procesos lo mandan a llamar
  int rank, size;//y obtinene estos valores
  
  /*Obtenemos nuestro ID*/
/*MPI_COM_WORLD es el grupo inicial (tiene todos los procesos) size es el numero del proceso y el rank puede ser 0-n (n numeero de procesos--ej lo corremos con 4 procesos entonces los esclavos 1-3) el 0 es maestro y el resto son esclavos*/
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);
  MPI_Comm_dup(MPI_COMM_WORLD, &new_comm);
  initialRank = rank;
  /*creamos nuestro propio MPI communicator (new_comm) y al inicio lo duplicamos con COMM_WORLD, esto ya que no podemos alterar el COMM_WORLD y lo usamos nosotros con MPI_Split para mantener solo los procesos vivos*/
  if( rank == 0 )
    master_process(rank, size);
  else
    slave_process(rank, size);
  
  MPI_Finalize();
  
  
  
  return 0;
}
