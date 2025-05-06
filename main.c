#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>
#include <petsc.h>

#include "AIFeatures.h" 

int main(int argc, char* argv[]) {
    struct timeval start_t, end_t;
    
    char* path = argv[1];
    // char* image_path = argv[2];
    char* feat_path = argv[2];
    MatrixFeatures mtx; 

    PetscErrorCode ierr;
    // // 初始化 PETSc 环境
    // ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);

    gettimeofday(&start_t, NULL);
    CreateMatrixFeaturesFromMTX(&mtx, path); 
    gettimeofday(&end_t, NULL);
    // printMtx(&mtx);
    // long long time_use = 1000000 * (end_t.tv_sec - start_t.tv_sec) + (end_t.tv_usec - start_t.tv_usec);
    // printf("CreateMatrixFeaturesFromMTX time: %lf s\n", (double)time_use / 1000000.0);
    SaveMatrixFeatures(&mtx,feat_path);
    // writeFeature(&mtx, feat_path); 
    // CreateMatrixFeaturesFromTXT(&mtx,path);
    // printMtx(&mtx);
    //  DestroyMatrixFeatures(&mtx);
    
    // testCreateMatrixFeaturesFromCOO();
    // testCreateMatrixFeaturesFromCSR();
    // CreateMatrixFeaturesFromTXT(&mtx, path);
    // printMtx(&mtx);

    // 结束 PETSc 环境
    // ierr = PetscFinalize(); CHKERRQ(ierr);
    return 0;
}