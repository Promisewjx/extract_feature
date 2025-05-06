#include "AIFeatures.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <stdbool.h>
#include <sys/time.h>
#include <inttypes.h>
#include "mmio.h"
#include <float.h>
#include <omp.h>

int CreateMatrixFeaturesFromMTX(MatrixFeatures* matrixFeatures, char* mtxFilePath){
    FunctionBegin;
    #ifndef COMPLEX

    FILE *file;
    int ret_code;
    MM_typecode mat_code;
    int is_integer = 0, is_real = 0, is_pattern = 0;
    int size = 128;

    // Initialize matrix features
    matrixFeatures->minNnzEachRow = DBL_MAX;
    matrixFeatures->maxNnzEachRow = 0;
    matrixFeatures->maxValue = -DBL_MAX;
    matrixFeatures->maxValueDiagonal = -DBL_MAX;
    matrixFeatures->trace = 0;
    matrixFeatures->traceAbs = 0;
    matrixFeatures->traceASquared = 0;
    matrixFeatures->arrNnzEachRow = 0; 
    matrixFeatures->avgDiagDist = 0.0;
    matrixFeatures->sigmaDiagDist = 0.0;

    // Open the matrix file
    if ((file = fopen(mtxFilePath, "r")) == NULL) {
        return 1; // 返回1表示失败
    }
    // Get the matrix type
    if (mm_read_banner(file, &mat_code) != 0) { 
        fclose(file);
        return 1;
    }
    // Check matrix type
    if (mm_is_complex(mat_code)) { 
        fclose(file);
        return 1; 
    } 
    if (mm_is_pattern(mat_code))  { is_pattern = 1; }
    if (mm_is_real(mat_code)) { is_real = 1; }
    if (mm_is_integer(mat_code)) { is_integer = 1; }

    // Check symmetry
    if (mm_is_symmetric(mat_code) || mm_is_hermitian(mat_code)) {
        matrixFeatures->isSymmetric = 1;
    } else {
        matrixFeatures->isSymmetric = 0;
    }

    // Get size of sparse matrix.
    ret_code = mm_read_mtx_crd_size(file, &matrixFeatures->rowNum, &matrixFeatures->colNum, &matrixFeatures->nnzMtx);

    if (ret_code != 0) {
        fclose(file);
        return 1;
    }
    if (is_real == 0 && is_integer == 0) {
        fclose(file);
        return 1;
    }   

    // Allocate memory for arrays
    int* nnzByRow = (int*)malloc(matrixFeatures->rowNum * sizeof(int64_t));
    double* rowSums = (double*)calloc(matrixFeatures->rowNum, sizeof(double));
    double* maxEachRow = (double*)calloc(matrixFeatures->rowNum, sizeof(double));
    double* minEachRow = (double*)malloc(matrixFeatures->rowNum * sizeof(double));
    double* maxEachCol = (double*)calloc(matrixFeatures->colNum, sizeof(double));
    double* minEachCol = (double*)malloc(matrixFeatures->colNum * sizeof(double));
    int* rowIndex = (int*)malloc(matrixFeatures->nnzMtx * sizeof(int64_t));
    int* colIndex = (int*)malloc(matrixFeatures->nnzMtx * sizeof(int64_t));
    double* diagonalElements = (double*)malloc(matrixFeatures->nnzMtx * sizeof(double));
    double* valArr = (double*)malloc(matrixFeatures->nnzMtx * sizeof(double));
    double *colSums = (double *)malloc(matrixFeatures->colNum * sizeof(double));
    double* rowDiffSum = (double*)calloc(matrixFeatures->rowNum, sizeof(double));
    double* colDiffSum = (double*)calloc(matrixFeatures->colNum, sizeof(double));

    // Allocate memory for image representation
    matrixFeatures->image = (int**)malloc(size * sizeof(int*));
    matrixFeatures->imageNnzDiagonal = (int**)malloc(size * sizeof(int*));
    matrixFeatures->imageRatio = (float**)malloc(size * sizeof(float*));
    matrixFeatures->imageMaxValue = (float**)malloc(size * sizeof(float*));
    matrixFeatures->imageMaxValueDia = (float**)malloc(size * sizeof(float*));
    float** imageAvg = (float**)malloc(size * sizeof(float*));
    int** imageBlocksize = (int**)malloc(size * sizeof(int*));

    for (int i = 0; i < size; i++) {
        matrixFeatures->image[i] = (int*)calloc(size, sizeof(int));
        matrixFeatures->imageNnzDiagonal[i] = (int*)calloc(size, sizeof(int));
        matrixFeatures->imageRatio[i] = (float*)calloc(size, sizeof(float));
        matrixFeatures->imageMaxValue[i] = (float*)calloc(size, sizeof(float));
        matrixFeatures->imageMaxValueDia[i] = (float*)calloc(size, sizeof(float));
        imageAvg[i] = (float*)calloc(size, sizeof(float));
        imageBlocksize[i] = (int*)calloc(size, sizeof(int));
    }

    if (!nnzByRow || !rowSums || !maxEachRow || 
        !minEachRow || !maxEachCol || !minEachCol || !rowIndex ||
        !colIndex || !diagonalElements || !valArr || !colSums ||
        !matrixFeatures->image || !matrixFeatures->imageNnzDiagonal ||
        !matrixFeatures->imageRatio || !imageAvg || !matrixFeatures->imageMaxValue ||
        !matrixFeatures->imageMaxValueDia || !imageBlocksize) {
        // If the assignment fails, exit the program
        perror("Memory allocation failed : 1");
        exit(EXIT_FAILURE);
    }
    memset(colSums, 0, matrixFeatures->colNum * sizeof(double));
    for(int i = 0;i < matrixFeatures->rowNum;i++){
        nnzByRow[i] = 0;
        minEachRow[i] = 1.e+5; 
    }
    for (int i = 0; i < matrixFeatures->colNum; ++i) 
        minEachCol[i] = 1.e+5;

    int normalBlockSizeLen = matrixFeatures->rowNum / size;
    int normalBlockSizeLeft = matrixFeatures->rowNum - normalBlockSizeLen * size;
    int normalBlockSizeLeftNn = normalBlockSizeLeft * normalBlockSizeLeft;
    int normalBlockSizeLeftN1 = normalBlockSizeLeft * normalBlockSizeLen;

    if(normalBlockSizeLen == 0)
        normalBlockSizeLen = 1;
    if(normalBlockSizeLeft == 0)
        normalBlockSizeLeft = 1;
    if(normalBlockSizeLeftNn == 0)
        normalBlockSizeLeftNn = 1;
    if(normalBlockSizeLeftN1 == 0)
        normalBlockSizeLeftN1 = 1;

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (i != size - 1 && j != size - 1) {
                imageBlocksize[i][j] = normalBlockSizeLen;
            } else if (i != size - 1 || j != size - 1) {
                imageBlocksize[i][j] = normalBlockSizeLeftN1;
            } else {
                imageBlocksize[i][j] = normalBlockSizeLeftNn;
            }
        }
    }

    // Read matrix elements
    int rowIdx, colIdx;
    double value,valueAbs;
    // double diagonalSum = 0.0;
    int positiveDiagCount = 0, negativeDiagCount = 0, positiveCount = 0;
    int nnzDiagonal = 0, nnzLower = 0, nnzUpper = 0;
    double normF = 0.0, norm1 = 0.0, normInf_ = 0.0;
    int upperBandWidth = 0,lowerBandWidth = 0;
    double trace = 0.0, traceAbs = 0.0, traceASquared = 0.0;
    // int symmetricCount = 0;
    // int valueSymmetricCount = 0;
    // double symmetrySnorm = 0.0; 
    // double symmetryAnorm = 0.0;
    // double symmetryFnorm = 0.0;
    // double symmetryFanorm = 0.0;

    for (int i = 0; i < matrixFeatures->nnzMtx; ++i) {
        if (fscanf(file, "%d %d %lf\n", &rowIdx, &colIdx, &value) != 3) {
            fprintf(stderr, "Error reading from file.\n");
            fclose(file);
            // return 1;
        }

        double valueAbs = fabs(value);
        
        double threshold = 1e-12;
        if(valueAbs < threshold)
            continue;
        
        rowIdx--;
        colIdx--;

        nnzByRow[rowIdx]++;
        colSums[colIdx] += valueAbs;
        rowSums[rowIdx] += valueAbs;
        normF += value * value;

        if (rowIdx == colIdx) {
            trace += value;
            traceAbs += valueAbs;
            // diagonalSum += valueAbs;
            if (value > 0){
                positiveDiagCount++;
            }else if (value < 0){
                negativeDiagCount++;
            } 
            nnzDiagonal++;
            diagonalElements[rowIdx] = value;
        }

        // Deal with symmetry
        if (matrixFeatures->isSymmetric) {
            if (rowIdx != colIdx) {
                nnzByRow[colIdx]++;
                nnzLower++;
                nnzUpper++;
                if (matrixFeatures->maxValue < value) {
                    matrixFeatures->maxValue = value;
                }
                if(value > 0){
                    positiveCount++;
                    positiveCount++;
                }
                colSums[rowIdx] += valueAbs;
                
                rowSums[colIdx] += valueAbs;

                normF += value * value;

                int bandWidth = abs(colIdx - rowIdx);
                if(bandWidth > upperBandWidth || bandWidth > lowerBandWidth){
                    upperBandWidth = bandWidth;
                    lowerBandWidth = bandWidth;
                }
            } else {
                if (matrixFeatures->maxValueDiagonal < value) {
                    matrixFeatures->maxValueDiagonal = value;
                }
                if(value > 0){
                    positiveCount++;
                }
            }
            // Calculate the maximum and minimum values for the log-scaled rows and columns of the matrix
            if (maxEachRow[rowIdx] < log10(valueAbs)) {
                maxEachRow[rowIdx] = log10(valueAbs);
            }
            if (valueAbs > 0.0 && minEachRow[rowIdx] > log10(valueAbs)) {
                minEachRow[rowIdx] = log10(valueAbs);
            }
            if (maxEachRow[colIdx] < log10(valueAbs)) {
                maxEachRow[colIdx] = log10(valueAbs);
            }
            if (valueAbs > 0.0 && minEachRow[colIdx] > log10(valueAbs)) {
                minEachRow[colIdx] = log10(valueAbs);
            }
            if (maxEachCol[colIdx] < log10(valueAbs)) {
                maxEachCol[colIdx] = log10(valueAbs);
            }
            if (valueAbs > 0.0 && minEachCol[colIdx] > log10(valueAbs)) {
                minEachCol[colIdx] = log10(valueAbs);
            }
            if (maxEachCol[rowIdx] < log10(valueAbs)) {
                maxEachCol[rowIdx] = log10(valueAbs);
            }
            if (valueAbs > 0.0 && minEachCol[rowIdx] > log10(valueAbs)) {
                minEachCol[rowIdx] = log10(valueAbs);
            }
        } 
        // Deal with asymmetry
        else {
            if(value > 0)
                positiveCount++;
            if (rowIdx > colIdx) {
                int bandWidth = rowIdx - colIdx;
                if (bandWidth > lowerBandWidth) {
                    lowerBandWidth = bandWidth;
                }
                nnzLower++;
                if (matrixFeatures->maxValue < value) {
                    matrixFeatures->maxValue = value;
                }
            } else if (rowIdx < colIdx) {
                int bandWidth = colIdx - rowIdx;
                if (bandWidth > upperBandWidth) {
                    upperBandWidth = bandWidth;
                }
                nnzUpper++;
                if (matrixFeatures->maxValue < value) {
                    matrixFeatures->maxValue = value;
                }
            } else {
                if (matrixFeatures->maxValueDiagonal < value) {
                    matrixFeatures->maxValueDiagonal = value;
                }
            }
            if (maxEachRow[rowIdx] < log10(valueAbs)) {
                maxEachRow[rowIdx] = log10(valueAbs);
            }
            if (valueAbs > 0.0 && minEachRow[rowIdx] > log10(valueAbs)) {
                minEachRow[rowIdx] = log10(valueAbs);
            }
            if (maxEachCol[colIdx] < log10(valueAbs)) {
                maxEachCol[colIdx] = log10(valueAbs);
            }
            if (valueAbs > 0.0 && minEachCol[colIdx] > log10(valueAbs)) {
                minEachCol[colIdx] = log10(valueAbs);
            }
        }

        if (matrixFeatures->rowNum <= size) {
            matrixFeatures->image[rowIdx][colIdx]++;
            matrixFeatures->imageRatio[rowIdx][colIdx]++;
            if (matrixFeatures->isSymmetric && (rowIdx != colIdx)) {
                matrixFeatures->image[colIdx][rowIdx]++;
                matrixFeatures->imageRatio[colIdx][rowIdx]++;
            }
            // value diagonal (havn't consider symm)
            if (rowIdx == colIdx) {
                matrixFeatures->imageMaxValueDia[rowIdx][colIdx] = fmax(matrixFeatures->imageMaxValueDia[rowIdx][colIdx], valueAbs);
                matrixFeatures->imageNnzDiagonal[rowIdx][colIdx]++;
            } else {
                matrixFeatures->imageMaxValue[rowIdx][colIdx] = fmax(matrixFeatures->imageMaxValue[rowIdx][colIdx], valueAbs);
                if (matrixFeatures->isSymmetric) {
                    matrixFeatures->imageMaxValue[colIdx][rowIdx] = fmax(matrixFeatures->imageMaxValue[colIdx][rowIdx], valueAbs);
                }
            }
        } else {
            matrixFeatures->image[rowIdx * size / matrixFeatures->rowNum][colIdx * size / matrixFeatures->colNum]++;
            imageAvg[rowIdx * size / matrixFeatures->rowNum][colIdx * size / matrixFeatures->colNum] += valueAbs;
            if (matrixFeatures->isSymmetric && (rowIdx != colIdx)) {
                matrixFeatures->image[colIdx * size / matrixFeatures->colNum][rowIdx * size / matrixFeatures->rowNum]++;
                imageAvg[colIdx * size / matrixFeatures->colNum][rowIdx * size / matrixFeatures->rowNum] += valueAbs;
            }
            // value diagonal (havn't consider symm)
            if (rowIdx == colIdx) {
                matrixFeatures->imageMaxValueDia[rowIdx * size / matrixFeatures->rowNum][colIdx * size / matrixFeatures->colNum] = fmax(matrixFeatures->imageMaxValueDia[rowIdx * size / matrixFeatures->rowNum][colIdx * size / matrixFeatures->colNum], valueAbs);
                matrixFeatures->imageNnzDiagonal[rowIdx * size / matrixFeatures->rowNum][colIdx * size / matrixFeatures->colNum]++;
            } else {
                matrixFeatures->imageMaxValue[rowIdx * size / matrixFeatures->rowNum][colIdx * size / matrixFeatures->colNum] = fmax(matrixFeatures->imageMaxValue[rowIdx * size / matrixFeatures->rowNum][colIdx * size / matrixFeatures->colNum], valueAbs);
                if (matrixFeatures->isSymmetric) {
                matrixFeatures->imageMaxValue[colIdx * size / matrixFeatures->colNum][rowIdx * size / matrixFeatures->rowNum] = fmax(matrixFeatures->imageMaxValue[colIdx * size / matrixFeatures->colNum][rowIdx * size / matrixFeatures->rowNum], valueAbs);
                }
            }
        }
    }
    // //   Find symmetry elements and calculate the symmetry norm
    // for (int i = 0; i < matrixFeatures->nnzMtx; ++i) {
    //     int rowIdx = rowIndex[i];
    //     int colIdx = colIndex[i];
    //     double value = valArr[i];
    //     double symmetric_value = matrix[colIdx][rowIdx];
    //     double diff = fabs(value - symmetric_value);

    //     symmetrySnorm += diff;
    //     symmetryAnorm = fmax(symmetryAnorm, diff);
    //     symmetryFnorm += diff * diff;
    //     symmetryFanorm += diff;
    //     symmetricCount++;

    //     if (fabs(value - symmetric_value) < DBL_EPSILON) {
    //         valueSymmetricCount++;
    //     }
    // }

    // // Calculate the symmetry norm
    // if (symmetricCount > 0) {
    //     symmetryFanorm /= symmetricCount;
    // }    
    // symmetryFnorm = sqrt(symmetryFnorm);  


    // Calculate the total number of non-zero elements
    matrixFeatures->nnzNum = nnzLower + nnzUpper + nnzDiagonal;
    // Calculate the fraction of positive values
    matrixFeatures->positiveFraction = (double)positiveCount / matrixFeatures->nnzNum;

    int sumRow = 0;
    double maxNnzEachRow = 0;
    double minNnzEachRow = DBL_MAX;
    double rowDivideMax = 0.0;
    double colDivideMax = 0.0;
    double tmp;
    int nonZeroRows = 0;
    double idx = 0.0, idxAll = 0.0;
    double arrNnzEachRow = 0.0;
    int diagonalDominantCount = 0;
    double nonDiagEleSum = 0.0;

    // Calculate the 1-norm (maximum column sum)
    for (int i = 0; i < matrixFeatures->colNum; ++i) {
        // Calculate norm1
        norm1 = fmax(norm1, colSums[i]);

        // 计算 colDivideMax
        if (minEachCol[i] != 0.0) {
            double tmp = maxEachCol[i] - minEachCol[i];
            colDivideMax = fmax(colDivideMax, tmp);
        }
    }

    for (int i = 0; i < matrixFeatures->rowNum; ++i) {
        // Calculate the infinity norm (maximum row sum) and nonZeroRows
        normInf_ = fmax(normInf_, rowSums[i]);
        if (nnzByRow[i] > 0) nonZeroRows++;

        // Calculate the maximum and minimum values for the number of non-zero elements per row, as well as the average value
        sumRow += nnzByRow[i];
        if (nnzByRow[i] > maxNnzEachRow) {
            maxNnzEachRow = nnzByRow[i];
        }
        if (nnzByRow[i] < minNnzEachRow) {
            minNnzEachRow = nnzByRow[i];
        }
        if (minEachRow[i] != 0.0) {
            double tmp = maxEachRow[i] - minEachRow[i];
            rowDivideMax = fmax(rowDivideMax, tmp);
        }

        // Calculate the ratio of diagonally dominant rows
        // idxAll = idxAll + 1.0;
        // if (rowSums[i] > 0) {
        //     idx = idx + 1.0;
        // }
        double diagAbs = fabs(diagonalElements[i]);
        // double sumWithoutDiag = rowSums[i] - diagAbs;
        // if (diagAbs > sumWithoutDiag) {
        //     diagonalDominantCount++;
        // }

        nonDiagEleSum += rowSums[i] - diagAbs;
    }

    matrixFeatures->averageNnzEachRow = 1.0 * sumRow / matrixFeatures->rowNum;
    
    for (int i = 0; i < matrixFeatures->rowNum; ++i) {
        double diff = nnzByRow[i] - matrixFeatures->averageNnzEachRow;
        arrNnzEachRow += diff * diff;
    }

    // Calculate the condition number (normInf / norm1)
    matrixFeatures->kappa = normInf_ / norm1;

    // Calculate the Frobenius norm
    matrixFeatures->normF = sqrt(normF);

    // Calculate the average and variance of diagonal elements and avgDiagDist and sigmaDiagDist
    double diagonalAverage = nnzDiagonal > 0 ? trace / nnzDiagonal : 0.0;
    double diagonalVariance = 0.0;
    double sumDiff = 0.0;
    double sumSquaredDiff = 0.0;

    for (int i = 0; i < nnzDiagonal; i++) {
        double diff = fabs(diagonalElements[i] - diagonalAverage);
        diagonalVariance += diff * diff;
        sumDiff += diff;
        sumSquaredDiff = diagonalVariance;
    }

    diagonalVariance /= nnzDiagonal;
    matrixFeatures->diagonalAverage = diagonalAverage;
    matrixFeatures->diagonalVariance = diagonalVariance;
    matrixFeatures->avgDiagDist = sumDiff / nnzDiagonal;
    matrixFeatures->sigmaDiagDist = sqrt(sumSquaredDiff / nnzDiagonal);

    matrixFeatures->nDummyRows = matrixFeatures->rowNum - nonZeroRows;

    // Calculate the sign of the diagonal elements (positive - negative)
    matrixFeatures->diagonalSign = positiveDiagCount - negativeDiagCount;

    matrixFeatures->trace = trace;
    matrixFeatures->traceAbs = traceAbs;
    matrixFeatures->traceASquared = traceAbs * traceAbs;
    matrixFeatures->nnzLower = nnzLower;
    matrixFeatures->nnzUpper = nnzUpper;
    matrixFeatures->nnzDiagonal = nnzDiagonal;
    // Calculate diagonal zero statistics
    matrixFeatures->diagZerostat = matrixFeatures->rowNum - nnzDiagonal;
    // Calculate diagonal certainty
    matrixFeatures->diagDefinite = (nnzDiagonal == matrixFeatures->rowNum);
    matrixFeatures->norm1 = norm1;
    matrixFeatures->normInf = normInf_;
    matrixFeatures->nnzRatio = 1.0 * matrixFeatures->nnzNum / (1.0 * matrixFeatures->rowNum * matrixFeatures->colNum);
    
    matrixFeatures->maxNnzEachRow = maxNnzEachRow;
    matrixFeatures->minNnzEachRow = minNnzEachRow;
    matrixFeatures->rowVariability = rowDivideMax;
    matrixFeatures->colVariability = colDivideMax;
    matrixFeatures->arrNnzEachRow = arrNnzEachRow / matrixFeatures->rowNum;
    // matrixFeatures->diagonalDominantRatio = (double)diagonalDominantCount / matrixFeatures->rowNum;
    if(nonDiagEleSum == 0)
        matrixFeatures->diagonalDominantRatio = 1.0;
    else
        matrixFeatures->diagonalDominantRatio = traceAbs / nonDiagEleSum;

    // matrixFeatures->symmetrySnorm = symmetrySnorm;
    // matrixFeatures->symmetryAnorm = symmetryAnorm;
    // matrixFeatures->symmetryFnorm = symmetryFnorm;
    // matrixFeatures->symmetryFanorm = symmetryFanorm;

    // Save bandwidth
    matrixFeatures->upBand = upperBandWidth;
    matrixFeatures->loBand = lowerBandWidth;
    // matrixFeatures->patternSymm = pattern_symm;
    // matrixFeatures->valueSymm = value_symm;

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrixFeatures->imageRatio[i][j] = 1.0 * matrixFeatures->image[i][j] / imageBlocksize[i][j];
            if (i != size - 1 && j != size - 1) {
                matrixFeatures->imageRatio[i][j] /= imageBlocksize[i][j];
            }
            if (imageAvg[i][j] != 0) {
                imageAvg[i][j] /= (1.0 * matrixFeatures->image[i][j]);
            }
        }
    }

    // Free up temporary memory
    free(nnzByRow);
    free(rowSums);
    free(maxEachRow);
    free(minEachRow);
    free(minEachCol);
    free(maxEachCol);
    free(rowIndex);
    free(colIndex);
    free(diagonalElements);
    free(valArr);
    free(colSums);
    free(rowDiffSum);
    free(colDiffSum);
    free(imageAvg);
    free(imageBlocksize);
    fclose(file);

    #else
        SETERRQ(PETSC_COMM_SELF,MM_UNSUPPORTED_TYPE,"Only support real matrix");
        FunctionReturn(MM_UNSUPPORTED_TYPE);
    #endif

    FunctionReturn(SUCCESS);
}



int CreateMatrixFeaturesFromCSR(MatrixFeatures* matrixFeatures, int nRow, int nCol, bool matSymType, bool matValueType, int* rowOffset, int* colIndex, double* value) {
    FunctionBegin;
    #ifndef COMPLEX
    // 如果matValueType为真，返回错误代码-1，因为当前只支持实数
    if (matValueType) {
        return -1; // 目前只支持实数
    }

    // 初始化矩阵特征结构体的行数和列数，以及是否对称
    matrixFeatures->rowNum = nRow;
    matrixFeatures->colNum = nCol;
    // 计算非零元素的数量
    long long nnz = rowOffset[nRow] - rowOffset[0];
    matrixFeatures->image=NULL;
    matrixFeatures->imageNnzDiagonal=NULL;
    matrixFeatures->imageRatio=NULL;
    matrixFeatures->imageMaxValue=NULL;
    matrixFeatures->imageMaxValueDia=NULL;
    // 初始化
    matrixFeatures->minNnzEachRow = DBL_MAX;
    matrixFeatures->maxNnzEachRow = 0;
    matrixFeatures->maxValue = -DBL_MAX;
    matrixFeatures->maxValueDiagonal = -DBL_MAX;
    matrixFeatures->trace = 0;
    matrixFeatures->traceAbs = 0;
    matrixFeatures->traceASquared = 0;
    matrixFeatures->arrNnzEachRow = 0;
    matrixFeatures->averageNnzEachRow = 0;
    matrixFeatures->diagZerostat = 0;
    // matrixFeatures->diagDefinite = 1;

    // 分配行和列的和、最大值、最小值的内存
    int* nnzByRow = (int*)malloc(matrixFeatures->rowNum * sizeof(int64_t));
    double* rowSums = (double*)calloc(nRow, sizeof(double));
    double* colSums = (double*)calloc(nCol, sizeof(double));
    double* rowMax = (double*)calloc(nRow, sizeof(double));
    double* colMax = (double*)calloc(nCol, sizeof(double));
    double* rowMin = (double*)calloc(nRow, sizeof(double));
    double* colMin = (double*)calloc(nCol, sizeof(double));
    double* maxEachRow = (double*)calloc(nRow, sizeof(double));
    double* minEachRow = (double*)malloc(nRow * sizeof(double));
    double* maxEachCol = (double*)calloc(nCol, sizeof(double));
    double* minEachCol = (double*)malloc(nCol * sizeof(double));
    double* diagonalElements = (double*)malloc(nnz * sizeof(double));

    // 如果内存分配失败，释放已分配的内存并返回错误代码-2
    if (!rowSums || !colSums || !rowMax || !colMax || !rowMin || !colMin
        || !nnzByRow || !maxEachRow || !diagonalElements || 
        !minEachRow || !maxEachCol || !minEachCol) {
        return -2; // 内存分配失败
    }

    memset(colSums, 0, matrixFeatures->colNum * sizeof(double));

    for(int i = 0;i < matrixFeatures->rowNum;i++){
        nnzByRow[i] = 0;
        minEachRow[i] = 1.e+5; 
    }
      
    for (int i = 0; i < matrixFeatures->colNum; ++i) 
        minEachCol[i] = 1.e+5;

    // 初始化各种范数和统计量
    double normF = 0.0;
    double norm1 = 0.0;
    double normInf = 0.0;
    // double symmetrySnorm = 0.0;
    // double symmetryAnorm = 0.0;
    // double symmetryFnorm = 0.0;
    double maxValue = -DBL_MAX;
    double maxValueDiagonal = -DBL_MAX;
    double trace = 0.0;
    double traceAbs = 0.0;
    int nnzDiagonal = 0;
    double sumDiagDist = 0.0;
    double sumDiagDistSquared = 0.0;
    int positiveDiagCount = 0;
    int diagZerostat = 0;
    int positiveCount = 0;
    int negativeDiagCount = 0;
    // int symmetricPairs = 0;
    // int valueSymmetricPairs = 0;
    int lowerBandWidth = 0;
    int upperBandWidth = 0;
    int nnzLower = 0;
    int nnzUpper = 0;

    int size = 128;
    // 动态分配内存和初始化其他数组
    matrixFeatures->image = (int**)malloc(size * sizeof(int*));
    for (int i = 0; i < size; ++i) {
        matrixFeatures->image[i] = (int*)malloc(size * sizeof(int));
        memset(matrixFeatures->image[i], 0, size * sizeof(int));
    }
    matrixFeatures->imageNnzDiagonal = (int**)malloc(size * sizeof(int*));
    for (int i = 0; i < size; ++i) {
        matrixFeatures->imageNnzDiagonal[i] = (int*)malloc(size * sizeof(int));
        memset(matrixFeatures->imageNnzDiagonal[i], 0, size * sizeof(int));
    }
    matrixFeatures->imageRatio = (float**)malloc(size * sizeof(float*));
    for (int i = 0; i < size; ++i) {
        matrixFeatures->imageRatio[i] = (float*)malloc(size * sizeof(float));
        memset(matrixFeatures->imageRatio[i], 0, size * sizeof(float));
    }
    float** imageAvg = (float**)malloc(size * sizeof(float*));
    for (int i = 0; i < size; ++i) {
        imageAvg[i] = (float*)malloc(size * sizeof(float));
        memset(imageAvg[i], 0, size * sizeof(float));
    }
    matrixFeatures->imageMaxValue = (float**)malloc(size * sizeof(float*));
    for (int i = 0; i < size; ++i) {
        matrixFeatures->imageMaxValue[i] = (float*)malloc(size * sizeof(float));
        memset(matrixFeatures->imageMaxValue[i], 0, size * sizeof(float));
    }
    matrixFeatures->imageMaxValueDia = (float**)malloc(size * sizeof(float*));
    for (int i = 0; i < size; ++i) {
        matrixFeatures->imageMaxValueDia[i] = (float*)malloc(size * sizeof(float));
        memset(matrixFeatures->imageMaxValueDia[i], 0, size * sizeof(float));
    }

    if (!matSymType) { // 如果不是对称压缩存储
        int isSymmetric = 1; // 假设矩阵对称
        const double tolerance = 1e-8; // 容差

        for (int i = 0; i < nRow && isSymmetric; ++i) {
            for (int idx = rowOffset[i]; idx < rowOffset[i + 1]; ++idx) {
                int j = colIndex[idx];     // 列索引
                double val_ij = value[idx]; // A(i, j)

                // 检查 A(j, i) 是否存在且相等
                bool found = false;
                for (int k = rowOffset[j]; k < rowOffset[j + 1]; ++k) {
                    if (colIndex[k] == i) {
                        double val_ji = value[k];
                        if (fabs(val_ij - val_ji) > tolerance) {
                            isSymmetric = false; // 不对称
                        }
                        found = true;
                        break;
                    }
                }

                if (!found) {
                    isSymmetric = 0; // A(j, i) 不存在，不对称
                }

                if (!isSymmetric) {
                    break; // 提前退出
                }
            }
        }

        // 更新对称性标志
        matrixFeatures->isSymmetric = isSymmetric;
    } else {
        matrixFeatures->isSymmetric = 1;
    }

    // Read Matrix.
    int normalBlockSizeLen = matrixFeatures->rowNum / size;
    int normalBlockSizeLeft = matrixFeatures->rowNum - normalBlockSizeLen * size;
    int normalBlockSizeLeftNn = normalBlockSizeLeft * normalBlockSizeLeft;
    int normalBlockSizeLeftN1 = normalBlockSizeLeft * normalBlockSizeLen;

    if(normalBlockSizeLen == 0)
        normalBlockSizeLen = 1;
    if(normalBlockSizeLeft == 0)
        normalBlockSizeLeft = 1;
    if(normalBlockSizeLeftNn == 0)
        normalBlockSizeLeftNn = 1;
    if(normalBlockSizeLeftN1 == 0)
        normalBlockSizeLeftN1 = 1;

    int** imageBlocksize = (int**)malloc(size * sizeof(int*));
    if (imageBlocksize == NULL) {
        fprintf(stderr, "Memory allocation failed for imageblocksize_\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < size; ++i) {
        imageBlocksize[i] = (int*)malloc(size * sizeof(int));
        if (imageBlocksize[i] == NULL) {
            fprintf(stderr, "Memory allocation failed for imageblocksize_[%d]\n", i);
            // 释放之前已分配的所有行
            for (int k = 0; k < i; ++k) {
                free(imageBlocksize[k]);
            }
            free(imageBlocksize);
            exit(EXIT_FAILURE);
        }
    }
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (i != size - 1 && j != size - 1) {
                imageBlocksize[i][j] = normalBlockSizeLen;
            } else if (i != size - 1 || j != size - 1) {
                imageBlocksize[i][j] = normalBlockSizeLeftN1;
            } else {
                imageBlocksize[i][j] = normalBlockSizeLeftNn;
            }
        }
    }

    for (int i = 0; i < nRow; ++i) {
        for (int j = rowOffset[i]; j < rowOffset[i + 1]; ++j) {
            int c = colIndex[j];
            double v = value[j];
            double valueAbs = fabs(v);

            double threshold = 1e-12;
            if(valueAbs < threshold)
                continue;

            nnzByRow[i]++;
            // 计算范数
            rowSums[i] += valueAbs;
            colSums[c] += valueAbs;

            if(v > 0)
                positiveCount++;

            // 最大值和最小值
            rowMax[i] = fmax(rowMax[i], v);
            colMax[c] = fmax(colMax[c], v);
            rowMin[i] = fmin(rowMin[i], v);
            colMin[c] = fmin(colMin[c], v);

            // Frobenius范数                                                   
            normF += v * v;

            int distance = abs(i - c);
            // 更新下带宽和上带宽
            if (i > c) {
                // 如果元素在对角线下方，更新下带宽
                lowerBandWidth = fmax(lowerBandWidth, distance);
                nnzLower++;
            } else if (i < c) {
                // 如果元素在对角线上方，更新上带宽
                upperBandWidth = fmax(upperBandWidth, distance);
                nnzUpper++;
            }

            // // 对称性范数
            // if (matSymType && i != c) {
            //     int symIndex = -1;
            //     for (int k = rowOffset[c]; k < rowOffset[c + 1]; ++k) {
            //         if (colIndex[k] == i) {
            //             symIndex = k;
            //             break;
            //         }
            //     }
            //     double symValue = symIndex != -1 ? value[symIndex] : 0.0;
            //     double diff = fabs(v - symValue);
            //     symmetrySnorm += diff;
            //     symmetryAnorm = fmax(symmetryAnorm, diff);
            //     symmetryFnorm += diff * diff;
            //     if (v == symValue) valueSymmetricPairs++;
            //     symmetricPairs++;
            // } else if (!matSymType) {
            //     symmetryFnorm += v * v;
            // }

            // 对角线元素
            if (i == c) {
                nnzDiagonal++;
                diagonalElements[i] = v;
                maxValueDiagonal = fmax(maxValueDiagonal, v);
                trace += v;
                traceAbs += fabs(v);
                if (v > 0) positiveDiagCount++;
                if (v == 0) diagZerostat++;
                // if (v <= 0) matrixFeatures->diagDefinite = 0;
                if (v < 0)
                    negativeDiagCount++;
            }

            // 更新最大值
            if(i != c)
                maxValue = fmax(maxValue, v);

            if (maxEachRow[i] < log10(valueAbs)) {
                maxEachRow[i] = log10(valueAbs);
            }
            if (valueAbs > 0.0 && minEachRow[i] > log10(valueAbs)) {
            minEachRow[i] = log10(valueAbs);
            }
            if (maxEachCol[c] < log10(valueAbs)) {
            maxEachCol[c] = log10(valueAbs);
            }
            if (valueAbs > 0.0 && minEachCol[c] > log10(valueAbs)) {
                minEachCol[c] = log10(valueAbs);
            }

            // 更新图像特征
            if (i < size && c < size) {
                matrixFeatures->image[i][c]++;
                matrixFeatures->imageRatio[i][c]++;
                if (matrixFeatures->isSymmetric && (i != c)) {
                    matrixFeatures->image[c][i]++;
                    matrixFeatures->imageRatio[c][i]++;
                }
                // 对角线上的值
                if (i == c) {
                    matrixFeatures->imageMaxValueDia[i][c] = fmax(matrixFeatures->imageMaxValueDia[i][c], valueAbs);
                    matrixFeatures->imageNnzDiagonal[i][c]++;
                } else {
                    matrixFeatures->imageMaxValue[i][c] = fmax(matrixFeatures->imageMaxValue[i][c], valueAbs);
                    if (matrixFeatures->isSymmetric) {
                        matrixFeatures->imageMaxValue[c][i] = fmax(matrixFeatures->imageMaxValue[c][i], valueAbs);
                    }
                }
            } else {
                int rowIdxScaled = i * size / matrixFeatures->rowNum;
                int colIdxScaled = c * size / matrixFeatures->colNum;
                matrixFeatures->image[rowIdxScaled][colIdxScaled]++;
                imageAvg[rowIdxScaled][colIdxScaled] += valueAbs;
                if (matrixFeatures->isSymmetric && (i != c)) {
                    matrixFeatures->image[colIdxScaled][rowIdxScaled]++;
                    imageAvg[colIdxScaled][rowIdxScaled] += valueAbs;
                }
                // 对角线上的值
                if (i == c) {
                    matrixFeatures->imageMaxValueDia[rowIdxScaled][colIdxScaled] = fmax(matrixFeatures->imageMaxValueDia[rowIdxScaled][colIdxScaled], valueAbs);
                    matrixFeatures->imageNnzDiagonal[rowIdxScaled][colIdxScaled]++;
                } else {
                    matrixFeatures->imageMaxValue[rowIdxScaled][colIdxScaled] = fmax(matrixFeatures->imageMaxValue[rowIdxScaled][colIdxScaled], valueAbs);
                    if (matrixFeatures->isSymmetric) {
                        matrixFeatures->imageMaxValue[colIdxScaled][rowIdxScaled] = fmax(matrixFeatures->imageMaxValue[colIdxScaled][rowIdxScaled], valueAbs);
                    }
                }
            }
        }
    }

    int sumRow = 0;
    double maxNnzEachRow = 0;
    double minNnzEachRow = DBL_MAX;
    double rowDivideMax = 0.0;
    double colDivideMax = 0.0;
    double tmp;
    int nonZeroRows = 0;
    // double idx = 0.0, idxAll = 0.0;
    double arrNnzEachRow = 0.0;
    int diagonalDominantCount = 0;
    double nonDiagEleSum = 0.0;

    // Calculate the 1-norm (maximum column sum)
    for (int i = 0; i < matrixFeatures->colNum; ++i) {
        // Calculate norm1
        norm1 = fmax(norm1, colSums[i]);

        // Calculate colDivideMax
        if (minEachCol[i] != 0.0) {
            double tmp = maxEachCol[i] - minEachCol[i];
            colDivideMax = fmax(colDivideMax, tmp);
        }
    }

    for (int i = 0; i < matrixFeatures->rowNum; ++i) {
        // Calculate the infinity norm (maximum row sum) and nonZeroRows
        normInf = fmax(normInf, rowSums[i]);
        if (nnzByRow[i] > 0) nonZeroRows++;

        // Calculate the maximum and minimum values for the number of non-zero elements per row, as well as the average value
        sumRow += nnzByRow[i];
        if (nnzByRow[i] > maxNnzEachRow) {
            maxNnzEachRow = nnzByRow[i];
        }
        if (nnzByRow[i] < minNnzEachRow) {
            minNnzEachRow = nnzByRow[i];
        }
        if (minEachRow[i] != 0.0) {
            double tmp = maxEachRow[i] - minEachRow[i];
            rowDivideMax = fmax(rowDivideMax, tmp);
        }

        // Calculate the ratio of diagonally dominant rows
        // idxAll = idxAll + 1.0;
        // if (rowSums[i] > 0) {
        //     idx = idx + 1.0;
        // }
        double diagAbs = fabs(diagonalElements[i]);
        // double sumWithoutDiag = rowSums[i] - diagAbs;
        // if (diagAbs > sumWithoutDiag) {
        //     diagonalDominantCount++;
        // }

        nonDiagEleSum += rowSums[i] - diagAbs;
    }

    matrixFeatures->averageNnzEachRow = 1.0 * sumRow / matrixFeatures->rowNum;

    for (int i = 0; i < nRow; ++i) {
        double diff = nnzByRow[i] - matrixFeatures->averageNnzEachRow;
        arrNnzEachRow += diff * diff;
    }

    matrixFeatures->arrNnzEachRow = arrNnzEachRow / nRow;
    if(nonDiagEleSum == 0)
        matrixFeatures->diagonalDominantRatio = 1.0;
    else
        matrixFeatures->diagonalDominantRatio = traceAbs / nonDiagEleSum;

    // Calculate the average and variance of diagonal elements and avgDiagDist and sigmaDiagDist
    double diagonalAverage = traceAbs / nnzDiagonal;
    double diagonalVariance = 0.0;
    double sumDiff = 0.0;
    double sumSquaredDiff = 0.0;

    for (int i = 0; i < nnzDiagonal; i++) {
        double diff = fabs(diagonalElements[i] - diagonalAverage);
        diagonalVariance += diff * diff;
        sumDiff += diff;
        sumSquaredDiff = diagonalVariance;
    }

    diagonalVariance /= nnzDiagonal;
    matrixFeatures->diagonalAverage = diagonalAverage;
    matrixFeatures->diagonalVariance = diagonalVariance;
    matrixFeatures->avgDiagDist = sumDiff / nnzDiagonal;
    matrixFeatures->sigmaDiagDist = sqrt(sumSquaredDiff / nnzDiagonal);

    // 计算各种范数
    matrixFeatures->normF = sqrt(normF);
    matrixFeatures->norm1 = norm1;
    matrixFeatures->normInf = normInf;
    // matrixFeatures->symmetrySnorm = symmetrySnorm;
    // matrixFeatures->symmetryAnorm = symmetryAnorm;
    // matrixFeatures->symmetryFnorm = sqrt(symmetryFnorm);
    // matrixFeatures->symmetryFanorm = symmetryFnorm / nnz;
    matrixFeatures->nnzUpper = nnzUpper;
    matrixFeatures->nnzLower = nnzLower;
    matrixFeatures->nnzDiagonal = nnzDiagonal;
    matrixFeatures->diagZerostat = diagZerostat;
    matrixFeatures->maxValue = maxValue;
    matrixFeatures->maxValueDiagonal = maxValueDiagonal;
    matrixFeatures->upBand = upperBandWidth;
    matrixFeatures->loBand = lowerBandWidth;
    matrixFeatures->diagonalSign = positiveDiagCount - negativeDiagCount;
    matrixFeatures->diagDefinite = (nnzDiagonal == matrixFeatures->rowNum);
    matrixFeatures->maxNnzEachRow = maxNnzEachRow;
    matrixFeatures->minNnzEachRow = minNnzEachRow;
    matrixFeatures->rowVariability = rowDivideMax;
    matrixFeatures->colVariability = colDivideMax;

    // 计算迹
    matrixFeatures->trace = trace;
    matrixFeatures->traceAbs = traceAbs;
    matrixFeatures->traceASquared = traceAbs * traceAbs;

    matrixFeatures->nnzNum = nnzLower + nnzUpper + nnzDiagonal;

    matrixFeatures->positiveFraction = (double)positiveCount / matrixFeatures->nnzNum;
    // 计算条件数
    matrixFeatures->kappa = normInf / norm1;
    matrixFeatures->nnzRatio = 1.0 * matrixFeatures->nnzNum / (1.0 * matrixFeatures->rowNum * matrixFeatures->colNum);

    // 计算对称性
    // matrixFeatures->patternSymm = matSymType ? (double)symmetricPairs / (double)nnz : 0.0;
    // matrixFeatures->valueSymm = matSymType ? (double)valueSymmetricPairs / (double)nnz : 0.0;

    // 计算虚拟行数
    matrixFeatures->nDummyRows = matrixFeatures->rowNum - nonZeroRows;

    // matrixFeatures->relSymm = (double)symmetricPairs / (double)(nnz - nnzDiagonal);

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrixFeatures->imageRatio[i][j] = 1.0 * matrixFeatures->image[i][j] / imageBlocksize[i][j];
            if (i != size - 1 && j != size - 1) {
                matrixFeatures->imageRatio[i][j] /= imageBlocksize[i][j];
            }
            if (imageAvg[i][j] != 0) {
                imageAvg[i][j] /= (1.0 * matrixFeatures->image[i][j]);
            }
        }
    }

    // 释放内存
    free(rowSums);
    free(colSums);
    free(rowMax);
    free(colMax);
    free(rowMin);
    free(colMin);
    free(maxEachRow);
    free(minEachRow);
    free(minEachCol);
    free(maxEachCol);
    free(diagonalElements);
    
    #else
        SETERRQ(PETSC_COMM_SELF,MM_UNSUPPORTED_TYPE,"Only support real matrix");
        FunctionReturn(MM_UNSUPPORTED_TYPE);
    #endif

    FunctionReturn(SUCCESS);

}

int CreateMatrixFeaturesFromCOO(MatrixFeatures* matrixFeatures, int nRow, int nCol, int nnz, bool matSymType, bool matValueType, int* rowId, int* colId, double* value) {
    FunctionBegin;
    #ifndef COMPLEX
    // 如果matValueType为true，表示矩阵包含非实数值，当前函数只支持实数值，因此返回错误码-1
    if (matValueType) {
        return -1; // Currently only supports real numbers
    }

    double diagonalSum = 0.0;
    int symmetricCount = 0;
    int valueSymmetricCount = 0;
    double normF = 0.0;
    double norm1 = 0.0;
    double normInf_ = 0.0;
    // double symmetrySnorm = 0.0; 
    // double symmetryAnorm = 0.0;
    // double symmetryFnorm = 0.0;
    // double symmetryFanorm = 0.0;
    int upperBandWidth = 0;  // 上带宽
    int lowerBandWidth = 0;  // 下带宽
    // 计算对角线符号
    int positiveDiagCount = 0;
    int negativeDiagCount = 0;
    int positiveCount = 0;
    int nnzDiagonal = 0;
    int nnzLower = 0;
    int nnzUpper = 0;
    double trace = 0.0, traceAbs = 0.0, traceASquared = 0.0;

    // 初始化矩阵特征结构体的基本属性
    matrixFeatures->rowNum = nRow;
    matrixFeatures->colNum = nCol;
    matrixFeatures->isSymmetric = matSymType ? 1 : 0; // 矩阵是否对称
    matrixFeatures->minNnzEachRow = DBL_MAX;
    matrixFeatures->maxNnzEachRow = 0;
    matrixFeatures->maxValue = -DBL_MAX;
    matrixFeatures->maxValueDiagonal = -DBL_MAX;
    matrixFeatures->trace = 0;
    matrixFeatures->traceAbs = 0;
    matrixFeatures->traceASquared = 0;
    matrixFeatures->arrNnzEachRow = 0;

    // 动态分配内存和初始化数组
    int* nnzByRow = (int*)malloc(matrixFeatures->rowNum * sizeof(int64_t));
    double* rowSums = (double*)calloc(matrixFeatures->rowNum, sizeof(double));
    double* maxEachRow = (double*)calloc(matrixFeatures->rowNum, sizeof(double));
    double* minEachRow = (double*)malloc(matrixFeatures->rowNum * sizeof(double));
    double* maxEachCol = (double*)calloc(matrixFeatures->colNum, sizeof(double));
    double* minEachCol = (double*)malloc(matrixFeatures->colNum * sizeof(double));
    double* diagonalElements = (double*)malloc(nnz * sizeof(double));
    double *colSums = (double *)malloc(matrixFeatures->colNum * sizeof(double));
    // 用于存储每行和每列的对称性差异
    double* rowDiffSum = (double*)calloc(matrixFeatures->rowNum, sizeof(double));
    double* colDiffSum = (double*)calloc(matrixFeatures->colNum, sizeof(double));

    if (!nnzByRow || !rowSums || !maxEachRow || 
        !minEachRow || !maxEachCol || !minEachCol || !diagonalElements || !colSums
        || !rowDiffSum || !colDiffSum) {
        // 如果分配失败，退出程序
        perror("Memory allocation failed : 1");
        
        exit(EXIT_FAILURE);
    }

    for(int i = 0;i < matrixFeatures->rowNum;i++){
        nnzByRow[i] = 0;
        minEachRow[i] = 1.e+5; 
    }
        
    for (int i = 0; i < matrixFeatures->colNum; ++i) 
        minEachCol[i] = 1.e+5;

    matrixFeatures->minNnzEachRow = DBL_MAX;
    memset(colSums, 0, matrixFeatures->colNum * sizeof(double));

    // double **matrix = (double **)malloc(matrixFeatures->rowNum * sizeof(double *));
    // for (int i = 0; i < matrixFeatures->rowNum; ++i) {
    //     matrix[i] = (double *)calloc(matrixFeatures->colNum, sizeof(double));
    //     if (matrix[i] == NULL) {
    //         // 处理内存分配失败
    //         fprintf(stderr, "Memory allocation failed for matrix row %d\n", i);
    //         // 释放之前已分配的所有行
    //         for (int k = 0; k < i; ++k) {
    //             free(matrix[k]);
    //         }
    //         free(matrix);
    //         return 1; // 或者根据函数的设计返回相应的错误代码
    //     }
    // }

    int size = 128;
    // 动态分配内存和初始化其他数组
    matrixFeatures->image = (int**)malloc(size * sizeof(int*));
    for (int i = 0; i < size; ++i) {
        matrixFeatures->image[i] = (int*)malloc(size * sizeof(int));
        memset(matrixFeatures->image[i], 0, size * sizeof(int));
    }
    matrixFeatures->imageNnzDiagonal = (int**)malloc(size * sizeof(int*));
    for (int i = 0; i < size; ++i) {
        matrixFeatures->imageNnzDiagonal[i] = (int*)malloc(size * sizeof(int));
        memset(matrixFeatures->imageNnzDiagonal[i], 0, size * sizeof(int));
    }
    matrixFeatures->imageRatio = (float**)malloc(size * sizeof(float*));
    for (int i = 0; i < size; ++i) {
        matrixFeatures->imageRatio[i] = (float*)malloc(size * sizeof(float));
        memset(matrixFeatures->imageRatio[i], 0, size * sizeof(float));
    }
    float** imageAvg = (float**)malloc(size * sizeof(float*));
    for (int i = 0; i < size; ++i) {
        imageAvg[i] = (float*)malloc(size * sizeof(float));
        memset(imageAvg[i], 0, size * sizeof(float));
    }
    matrixFeatures->imageMaxValue = (float**)malloc(size * sizeof(float*));
    for (int i = 0; i < size; ++i) {
        matrixFeatures->imageMaxValue[i] = (float*)malloc(size * sizeof(float));
        memset(matrixFeatures->imageMaxValue[i], 0, size * sizeof(float));
    }
    matrixFeatures->imageMaxValueDia = (float**)malloc(size * sizeof(float*));
    for (int i = 0; i < size; ++i) {
        matrixFeatures->imageMaxValueDia[i] = (float*)malloc(size * sizeof(float));
        memset(matrixFeatures->imageMaxValueDia[i], 0, size * sizeof(float));
    }

     // Read Matrix.
    int normalBlockSizeLen = matrixFeatures->rowNum / size;
    int normalBlockSizeLeft = matrixFeatures->rowNum - normalBlockSizeLen * size;
    int normalBlockSizeLeftNn = normalBlockSizeLeft * normalBlockSizeLeft;
    int normalBlockSizeLeftN1 = normalBlockSizeLeft * normalBlockSizeLen;

    if(normalBlockSizeLen == 0)
        normalBlockSizeLen = 1;
    if(normalBlockSizeLeft == 0)
        normalBlockSizeLeft = 1;
    if(normalBlockSizeLeftNn == 0)
        normalBlockSizeLeftNn = 1;
    if(normalBlockSizeLeftN1 == 0)
        normalBlockSizeLeftN1 = 1;

    int** imageBlocksize = (int**)malloc(size * sizeof(int*));
    if (imageBlocksize == NULL) {
        fprintf(stderr, "Memory allocation failed for imageblocksize_\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < size; ++i) {
        imageBlocksize[i] = (int*)malloc(size * sizeof(int));
        if (imageBlocksize[i] == NULL) {
            fprintf(stderr, "Memory allocation failed for imageblocksize_[%d]\n", i);
            // 释放之前已分配的所有行
            for (int k = 0; k < i; ++k) {
                free(imageBlocksize[k]);
            }
            free(imageBlocksize);
            exit(EXIT_FAILURE);
        }
    }
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (i != size - 1 && j != size - 1) {
                imageBlocksize[i][j] = normalBlockSizeLen;
            } else if (i != size - 1 || j != size - 1) {
                imageBlocksize[i][j] = normalBlockSizeLeftN1;
            } else {
                imageBlocksize[i][j] = normalBlockSizeLeftNn;
            }
        }
    }
    for (int i = 0; i < nnz; ++i) {
        int r = rowId[i];
        int c = colId[i];
        double v = value[i];
        double valueAbs = fabs(v);

        double threshold = 1e-12;
        if(fabs(v) < threshold)
            continue;
        
        nnzByRow[r]++;
        // 更新列和
        colSums[c] += valueAbs;
        // 更新 Frobenius 范数
        normF += v * v;
        // 计算无穷范数（行和的最大值）
        rowSums[r] += valueAbs;

        if (r == c) {
            trace += v;
            traceAbs += valueAbs;
            if (v > 0) {
                positiveDiagCount++;
            } else if (v < 0) {
                negativeDiagCount++;
            }
            nnzDiagonal++;
            diagonalElements[r] = v;
        }

        // 处理对称
        if (matrixFeatures->isSymmetric) {
            if (r != c) {
                nnzByRow[c]++;
                nnzLower++;
                nnzUpper++;
                if (matrixFeatures->maxValue < v) {
                    matrixFeatures->maxValue = v;
                }

                colSums[r] += valueAbs;
                normF += v * v;
                rowSums[c] += valueAbs;

                int bandWidth = abs(c - r);
                if(bandWidth > upperBandWidth || bandWidth > lowerBandWidth){
                    upperBandWidth = bandWidth;
                    lowerBandWidth = bandWidth;
                }
                if(v > 0){
                    positiveCount++;
                    positiveCount++;
                }

            } else {
                if (matrixFeatures->maxValueDiagonal < v) {
                    matrixFeatures->maxValueDiagonal = v;
                }
                if(v > 0){
                    positiveCount++;
                }
            }
            // own row-v
            if (maxEachRow[r] < log10(valueAbs)) {
                maxEachRow[r] = log10(valueAbs);
            }
            if (valueAbs > 0.0 && minEachRow[r] > log10(valueAbs)) {
            minEachRow[r] = log10(valueAbs);
            }
            if (maxEachRow[c] < log10(valueAbs)) {
                maxEachRow[c] = log10(valueAbs);
            }
            if (valueAbs > 0.0 && minEachRow[c] > log10(valueAbs)) {
                minEachRow[c] = log10(valueAbs);
            }
            if (maxEachCol[c] < log10(valueAbs)) {
            maxEachCol[c] = log10(valueAbs);
            }
            if (valueAbs > 0.0 && minEachCol[c] > log10(valueAbs)) {
                minEachCol[c] = log10(valueAbs);
            }
            if (maxEachCol[r] < log10(valueAbs)) {
                maxEachCol[r] = log10(valueAbs);
            }
            if (valueAbs > 0.0 && minEachCol[r] > log10(valueAbs)) {
                minEachCol[r] = log10(valueAbs);
            }
        } 
        // 处理非对称
        else {
            if(v > 0)
                positiveCount++;
            if (r > c) {
                int bandWidth = r - c;
                if (bandWidth > lowerBandWidth) {
                    lowerBandWidth = bandWidth;
                }
                nnzLower++;
                if (matrixFeatures->maxValue < v) {
                    matrixFeatures->maxValue = v;
                }
            } else if (r < c) {
                int bandWidth = c - r;
                if (bandWidth > upperBandWidth) {
                    upperBandWidth = bandWidth;
                }
                nnzUpper++;
                if (matrixFeatures->maxValue < v) {
                    matrixFeatures->maxValue = v;
                }
            } else {
                if (matrixFeatures->maxValueDiagonal < v) {
                    matrixFeatures->maxValueDiagonal = v;
                }
            }
            if (maxEachRow[r] < log10(valueAbs)) {
                maxEachRow[r] = log10(valueAbs);
            }
            if (valueAbs > 0.0 && minEachRow[r] > log10(valueAbs)) {
                minEachRow[r] = log10(valueAbs);
            }
            if (maxEachCol[c] < log10(valueAbs)) {
                maxEachCol[c] = log10(valueAbs);
            }
            if (valueAbs > 0.0 && minEachCol[c] > log10(valueAbs)) {
                minEachCol[c] = log10(valueAbs);
            }
        }

        if (matrixFeatures->rowNum <= size) {
            matrixFeatures->image[r][c]++;
            matrixFeatures->imageRatio[r][c]++;
            if (matrixFeatures->isSymmetric && (r != c)) {
                matrixFeatures->image[c][r]++;
                matrixFeatures->imageRatio[c][r]++;
            }
            // value diagonal (havn't consider symm)
            if (r == c) {
                
                matrixFeatures->imageMaxValueDia[r][c] = fmax(matrixFeatures->imageMaxValueDia[r][c], valueAbs);
                matrixFeatures->imageNnzDiagonal[r][c]++;
            } else {
                matrixFeatures->imageMaxValue[r][c] = fmax(matrixFeatures->imageMaxValue[r][c], valueAbs);
                if (matrixFeatures->isSymmetric) {
                    matrixFeatures->imageMaxValue[c][r] = fmax(matrixFeatures->imageMaxValue[c][r], valueAbs);
                }
            }
        } else {
            matrixFeatures->image[r * size / matrixFeatures->rowNum][c * size / matrixFeatures->colNum]++;
            imageAvg[r * size / matrixFeatures->rowNum][c * size / matrixFeatures->colNum] += valueAbs;
            if (matrixFeatures->isSymmetric && (r != c)) {
                matrixFeatures->image[c * size / matrixFeatures->colNum][r * size / matrixFeatures->rowNum]++;
                imageAvg[c * size / matrixFeatures->colNum][r * size / matrixFeatures->rowNum] += valueAbs;
            }
            // value diagonal (havn't consider symm)
            if (r == c) {
                matrixFeatures->imageMaxValueDia[r * size / matrixFeatures->rowNum][c * size / matrixFeatures->colNum] = fmax(matrixFeatures->imageMaxValueDia[r * size / matrixFeatures->rowNum][c * size / matrixFeatures->colNum], valueAbs);
                matrixFeatures->imageNnzDiagonal[r * size / matrixFeatures->rowNum][c * size / matrixFeatures->colNum]++;
            } else {
                matrixFeatures->imageMaxValue[r * size / matrixFeatures->rowNum][c * size / matrixFeatures->colNum] = fmax(matrixFeatures->imageMaxValue[r * size / matrixFeatures->rowNum][c * size / matrixFeatures->colNum], valueAbs);
                if (matrixFeatures->isSymmetric) {
                matrixFeatures->imageMaxValue[c * size / matrixFeatures->colNum][r * size / matrixFeatures->rowNum] = fmax(matrixFeatures->imageMaxValue[c * size / matrixFeatures->colNum][r * size / matrixFeatures->rowNum], valueAbs);
                }
            }
        }
    }
    
    matrixFeatures->nnzNum = nnzLower + nnzUpper + nnzDiagonal;

    // // 查找对称元素并计算对称性范数
    // for (int i = 0; i < matrixFeatures->nnzMtx; ++i) {
    //     int r = rowIndex[i];
    //     int c = colIndex[i];
    //     double value = valArr[i];
    //     double symmetric_value = matrix[c][r];
    //     double diff = fabs(value - symmetric_value);

    //     symmetrySnorm += diff;
    //     symmetryAnorm = fmax(symmetryAnorm, diff);
    //     symmetryFnorm += diff * diff;
    //     symmetryFanorm += diff;
    //     symmetricCount++;

    //     if (fabs(value - symmetric_value) < DBL_EPSILON) {
    //         valueSymmetricCount++;
    //     }
    // }

    // // 计算对称性范数
    // if (symmetricCount > 0) {
    //     symmetryFanorm /= symmetricCount;
    // }      

    int sumRow = 0;
    double maxNnzEachRow = 0;
    double minNnzEachRow = DBL_MAX;
    double rowDivideMax = 0.0;
    double colDivideMax = 0.0;
    double tmp;
    int nonZeroRows = 0;
    double arrNnzEachRow = 0.0;
    int diagonalDominantCount = 0;
    double nonDiagEleSum = 0.0;

    // Calculate the 1-norm (maximum column sum)
    
    for (int i = 0; i < matrixFeatures->colNum; ++i) {
        // Calculate norm1
        norm1 = fmax(norm1, colSums[i]);

        // Calculate colDivideMax
        if (minEachCol[i] != 0.0) {
            double tmp = maxEachCol[i] - minEachCol[i];
            colDivideMax = fmax(colDivideMax, tmp);
        }
    }

    for (int i = 0; i < matrixFeatures->rowNum; ++i) {
        // Calculate the infinity norm (maximum row sum) and nonZeroRows
        normInf_ = fmax(normInf_, rowSums[i]);
        if (nnzByRow[i] > 0) nonZeroRows++;

        // Calculate the maximum and minimum values for the number of non-zero elements per row, as well as the average value
        sumRow += nnzByRow[i];
        if (nnzByRow[i] > maxNnzEachRow) {
            maxNnzEachRow = nnzByRow[i];
        }
        if (nnzByRow[i] < minNnzEachRow) {
            minNnzEachRow = nnzByRow[i];
        }
        if (minEachRow[i] != 0.0) {
            double tmp = maxEachRow[i] - minEachRow[i];
            rowDivideMax = fmax(rowDivideMax, tmp);
        }

        // Calculate the ratio of diagonally dominant rows
        // idxAll = idxAll + 1.0;
        // if (rowSums[i] > 0) {
        //     idx = idx + 1.0;
        // }
        double diagAbs = fabs(diagonalElements[i]);
        // double sumWithoutDiag = rowSums[i] - diagAbs;
        // if (diagAbs > sumWithoutDiag) {
        //     diagonalDominantCount++;
        // }

        nonDiagEleSum += rowSums[i] - diagAbs;
    }
    matrixFeatures->averageNnzEachRow = 1.0 * sumRow / matrixFeatures->rowNum;
    
    for (int i = 0; i < matrixFeatures->rowNum; ++i) {
        double diff = nnzByRow[i] - matrixFeatures->averageNnzEachRow;
        arrNnzEachRow += diff * diff;
    }
    // // 计算最终的Frobenius范数
    // symmetryFnorm = sqrt(symmetryFnorm);

    
    // Calculate the average and variance of diagonal elements and avgDiagDist and sigmaDiagDist
    double diagonalAverage = trace / nnzDiagonal;
    double diagonalVariance = 0.0;
    double sumDiff = 0.0;
    double sumSquaredDiff = 0.0;

    for (int i = 0; i < nnzDiagonal; i++) {
        double diff = fabs(diagonalElements[i] - diagonalAverage);
        diagonalVariance += diff * diff;
        sumDiff += diff;
        sumSquaredDiff = diagonalVariance;
    }

    diagonalVariance /= nnzDiagonal;
    matrixFeatures->diagonalAverage = diagonalAverage;
    matrixFeatures->diagonalVariance = diagonalVariance;
    matrixFeatures->avgDiagDist = sumDiff / nnzDiagonal;
    matrixFeatures->sigmaDiagDist = sqrt(sumSquaredDiff / nnzDiagonal);

    // double pattern_symm = 0.0;
    // double value_symm = 0.0;

    // // 计算 patternSymm 和 valueSymm
    // if(matrixFeatures->isSymmetric){
    //     pattern_symm = 1.0;
    //     value_symm = 1.0;
    // }else{
    //     if (matrixFeatures->nnzMtx > nnzDiagonal) {
    //         pattern_symm = (double)symmetricCount / (double)(matrixFeatures->nnzMtx - nnzDiagonal);
    //         value_symm = (double)valueSymmetricCount / (double)(matrixFeatures->nnzMtx - nnzDiagonal);
    //     }
    // }

    // 计算相对对称性relSymm
    // matrixFeatures->relSymm = (matrixFeatures->symmetryFnorm / normF) > 0 ? (matrixFeatures->symmetryFnorm / normF) : 0;

    // 计算虚拟行数
    matrixFeatures->nDummyRows = matrixFeatures->rowNum - nonZeroRows;

    matrixFeatures->diagonalSign = positiveDiagCount - negativeDiagCount;
    // 正样本比例
    matrixFeatures->positiveFraction = (double)positiveCount / matrixFeatures->nnzNum;

    // 计算Frobenius范数
    matrixFeatures->normF = sqrt(normF);
    matrixFeatures->trace = trace;
    matrixFeatures->traceAbs = traceAbs;
    matrixFeatures->traceASquared = traceAbs * traceAbs;
    matrixFeatures->nnzUpper = nnzUpper;
    matrixFeatures->nnzLower = nnzLower;
    matrixFeatures->nnzDiagonal = nnzDiagonal;
    // 计算对角线零统计
    matrixFeatures->diagZerostat = matrixFeatures->rowNum - nnzDiagonal;
    // 计算对角线确定性
    matrixFeatures->diagDefinite = (nnzDiagonal == matrixFeatures->rowNum);

    matrixFeatures->normInf = normInf_;

    // 条件数
    matrixFeatures->kappa = normInf_ / norm1;
    matrixFeatures->nnzRatio = 1.0 * matrixFeatures->nnzNum / (1.0 * matrixFeatures->rowNum * matrixFeatures->colNum);

    matrixFeatures->maxNnzEachRow = maxNnzEachRow;
    matrixFeatures->minNnzEachRow = minNnzEachRow;
    matrixFeatures->rowVariability = rowDivideMax;
    matrixFeatures->colVariability = colDivideMax;
    matrixFeatures->arrNnzEachRow = arrNnzEachRow / matrixFeatures->rowNum;
    if(nonDiagEleSum == 0)
        matrixFeatures->diagonalDominantRatio = 1.0;
    else
        matrixFeatures->diagonalDominantRatio = traceAbs / nonDiagEleSum;

    // 保存对称性范数到MTX结构体
    matrixFeatures->norm1 = norm1;
    // matrixFeatures->symmetrySnorm = symmetrySnorm;
    // matrixFeatures->symmetryAnorm = symmetryAnorm;
    // matrixFeatures->symmetryFnorm = symmetryFnorm;
    // matrixFeatures->symmetryFanorm = symmetryFanorm;
    // 保存带宽
    matrixFeatures->upBand = upperBandWidth;
    matrixFeatures->loBand = lowerBandWidth;
    // matrixFeatures->patternSymm = pattern_symm;
    // matrixFeatures->valueSymm = value_symm;
 
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrixFeatures->imageRatio[i][j] = 1.0 * matrixFeatures->image[i][j] / imageBlocksize[i][j];
            if (i != size - 1 && j != size - 1) {
                matrixFeatures->imageRatio[i][j] /= imageBlocksize[i][j];
            }
            if (imageAvg[i][j] != 0) {
                imageAvg[i][j] /= (1.0 * matrixFeatures->image[i][j]);
            }
        }
    }

    // 释放临时内存
    free(nnzByRow);
    free(rowSums);
    free(maxEachRow);
    free(minEachRow);
    free(minEachCol);
    free(maxEachCol);
    free(diagonalElements);
    free(colSums);
    free(rowDiffSum);
    free(colDiffSum);
    free(imageAvg);
    // free(matrix);

    #else
        SETERRQ(PETSC_COMM_SELF,MM_UNSUPPORTED_TYPE,"Only support real matrix");
        FunctionReturn(MM_UNSUPPORTED_TYPE);
    #endif

    FunctionReturn(SUCCESS);
}

int CreateMatrixFeaturesFromTXT(MatrixFeatures* matrixFeatures, char* featureFilePath) {
    FunctionBegin;
    #ifndef COMPLEX
    if (matrixFeatures == NULL || featureFilePath == NULL) {
        return -1; // 错误码1：无效的参数
    }

    matrixFeatures->image=NULL;
    matrixFeatures->imageNnzDiagonal=NULL;
    matrixFeatures->imageRatio=NULL;
    matrixFeatures->imageMaxValue=NULL;
    matrixFeatures->imageMaxValueDia=NULL;

    FILE* file = fopen(featureFilePath, "r");
    if (file == NULL) {
        return -2; // 错误码2：文件打开失败
    }

    int imageIndex = 0; // 用于跟踪当前正在读取哪个图像数据块
    void** currentDataPtr = NULL; // 当前正在读取的数据块的指针
    char line[1024];

    while (fgets(line, sizeof(line), file)) {
        if (sscanf(line, "rowNum %d", &matrixFeatures->rowNum) == 1) continue;
        if (sscanf(line, "colNum %d", &matrixFeatures->colNum) == 1) continue;
        if (sscanf(line, "nnz %d", &matrixFeatures->nnzNum) == 1) continue;
        if (sscanf(line, "nnzRatio %lf", &matrixFeatures->nnzRatio) == 1) continue;
        if (sscanf(line, "nnzLower %lld", &matrixFeatures->nnzLower) == 1) continue;
        if (sscanf(line, "nnzUpper %lld", &matrixFeatures->nnzUpper) == 1) continue;
        if (sscanf(line, "nnzDiagonal %lld", &matrixFeatures->nnzDiagonal) == 1) continue;
        if (sscanf(line, "averageNnzEachRow %lf", &matrixFeatures->averageNnzEachRow) == 1) continue;
        if (sscanf(line, "maxNnzEachRow %lf", &matrixFeatures->maxNnzEachRow) == 1) continue;
        if (sscanf(line, "arrNnzEachRow %lf", &matrixFeatures->arrNnzEachRow) == 1) continue;
        if (sscanf(line, "maxValue %lf", &matrixFeatures->maxValue) == 1) continue;
        if (sscanf(line, "maxValueDiagonal %lf", &matrixFeatures->maxValueDiagonal) == 1) continue;
        if (sscanf(line, "diagonalDominantRatio %lf", &matrixFeatures->diagonalDominantRatio) == 1) continue;
        if (sscanf(line, "isSymmetric %d", &matrixFeatures->isSymmetric) == 1) continue;
        // if (sscanf(line, "patternSymm %lf", &matrixFeatures->patternSymm) == 1) continue;
        // if (sscanf(line, "valueSymm %lf", &matrixFeatures->valueSymm) == 1) continue;
        if (sscanf(line, "rowVariability %lf", &matrixFeatures->rowVariability) == 1) continue;
        if (sscanf(line, "colVariability %lf", &matrixFeatures->colVariability) == 1) continue;
        if (sscanf(line, "trace %lf", &matrixFeatures->trace) == 1) continue;
        if (sscanf(line, "traceAbs %lf", &matrixFeatures->traceAbs) == 1) continue;
        if (sscanf(line, "traceASquared %lf", &matrixFeatures->traceASquared) == 1) continue;
        if (sscanf(line, "norm1 %lf", &matrixFeatures->norm1) == 1) continue;
        if (sscanf(line, "normInf %lf", &matrixFeatures->normInf) == 1) continue;
        if (sscanf(line, "normF %lf", &matrixFeatures->normF) == 1) continue;
        // if (sscanf(line, "symmetrySnorm %lf", &matrixFeatures->symmetrySnorm) == 1) continue;
        // if (sscanf(line, "symmetryAnorm %lf", &matrixFeatures->symmetryAnorm) == 1) continue;
        // if (sscanf(line, "symmetryFnorm %lf", &matrixFeatures->symmetryFnorm) == 1) continue;
        // if (sscanf(line, "symmetryFanorm %lf", &matrixFeatures->symmetryFanorm) == 1) continue;
        if (sscanf(line, "nDummyRows %d", &matrixFeatures->nDummyRows) == 1) continue;
        if (sscanf(line, "diagZerostat %d", &matrixFeatures->diagZerostat) == 1) continue;
        if (sscanf(line, "diagDefinite %d", &matrixFeatures->diagDefinite) == 1) continue;
        if (sscanf(line, "diagonalAverage %lf", &matrixFeatures->diagonalAverage) == 1) continue;
        if (sscanf(line, "diagonalVariance %lf", &matrixFeatures->diagonalVariance) == 1) continue;
        if (sscanf(line, "diagonalSign %d", &matrixFeatures->diagonalSign) == 1) continue;
        if (sscanf(line, "upBand %d", &matrixFeatures->upBand) == 1) continue;
        if (sscanf(line, "loBand %d", &matrixFeatures->loBand) == 1) continue;
        if (sscanf(line, "avgDiagDist %lf", &matrixFeatures->avgDiagDist) == 1) continue;
        if (sscanf(line, "sigmaDiagDist %lf", &matrixFeatures->sigmaDiagDist) == 1) continue;
        // if (sscanf(line, "relSymm %lf", &matrixFeatures->relSymm) == 1) continue;
        if (sscanf(line, "kappa %lf", &matrixFeatures->kappa) == 1) continue;
        if (sscanf(line, "positiveFraction %lf", &matrixFeatures->positiveFraction) == 1) continue;

        // 检查是否是新的图像数据块的开始
        if (strncmp(line, "image", 5) == 0) {
            // 确定当前数据块类型并分配内存
            int** intData = NULL;
            float** floatData = NULL;
            switch (imageIndex) {
                case 0:
                    matrixFeatures->image = (int**)malloc(128 * sizeof(int*));
                    if (matrixFeatures->image == NULL) {
                        fclose(file);
                        return -3; // 错误码3：内存分配失败
                    }
                    for (int i = 0; i < 128; ++i) {
                        matrixFeatures->image[i] = (int*)malloc(128 * sizeof(int));
                        if (matrixFeatures->image[i] == NULL) {
                            // 释放已分配的内存
                            for (int j = 0; j < i; ++j) {
                                free(matrixFeatures->image[j]);
                            }
                            free(matrixFeatures->image);
                            fclose(file);
                            return -3; // 错误码3：内存分配失败
                        }
                    }
                    break;
                case 1:
                    matrixFeatures->imageNnzDiagonal = (int**)malloc(128 * sizeof(int*));
                    if (matrixFeatures->imageNnzDiagonal == NULL) {
                        fclose(file);
                        return -3; // 错误码3：内存分配失败
                    }
                    for (int i = 0; i < 128; ++i) {
                        matrixFeatures->imageNnzDiagonal[i] = (int*)malloc(128 * sizeof(int));
                        if (matrixFeatures->imageNnzDiagonal[i] == NULL) {
                            // 释放已分配的内存
                            for (int j = 0; j < i; ++j) {
                                free(matrixFeatures->imageNnzDiagonal[j]);
                            }
                            free(matrixFeatures->imageNnzDiagonal);
                            fclose(file);
                            return -3; // 错误码3：内存分配失败
                        }
                    }
                    break;
                case 2:
                    matrixFeatures->imageRatio = (float**)malloc(128 * sizeof(float*));
                    if (matrixFeatures->imageRatio == NULL) {
                        fclose(file);
                        return -3; // 错误码3：内存分配失败
                    }
                    for (int i = 0; i < 128; ++i) {
                        matrixFeatures->imageRatio[i] = (float*)malloc(128 * sizeof(float));
                        if (matrixFeatures->imageRatio[i] == NULL) {
                            // 释放已分配的内存
                            for (int j = 0; j < i; ++j) {
                                free(matrixFeatures->imageRatio[j]);
                            }
                            free(matrixFeatures->imageRatio);
                            fclose(file);
                            return -3; // 错误码3：内存分配失败
                        }
                    }
                    break;
                case 3:
                    matrixFeatures->imageMaxValue = (float**)malloc(128 * sizeof(float*));
                    if (matrixFeatures->imageMaxValue == NULL) {
                        fclose(file);
                        return -3; // 错误码3：内存分配失败
                    }
                    for (int i = 0; i < 128; ++i) {
                        matrixFeatures->imageMaxValue[i] = (float*)malloc(128 * sizeof(float));
                        if (matrixFeatures->imageMaxValue[i] == NULL) {
                            // 释放已分配的内存
                            for (int j = 0; j < i; ++j) {
                                free(matrixFeatures->imageMaxValue[j]);
                            }
                            free(matrixFeatures->imageMaxValue);
                            fclose(file);
                            return -3; // 错误码3：内存分配失败
                        }
                    }
                    break;
                case 4:
                    matrixFeatures->imageMaxValueDia = (float**)malloc(128 * sizeof(float*));
                    if (matrixFeatures->imageMaxValueDia == NULL) {
                        fclose(file);
                        return -3; // 错误码3：内存分配失败
                    }
                    for (int i = 0; i < 128; ++i) {
                        matrixFeatures->imageMaxValueDia[i] = (float*)malloc(128 * sizeof(float));
                        if (matrixFeatures->imageMaxValueDia[i] == NULL) {
                            // 释放已分配的内存
                            for (int j = 0; j < i; ++j) {
                                free(matrixFeatures->imageMaxValueDia[j]);
                            }
                            free(matrixFeatures->imageMaxValueDia);
                            fclose(file);
                            return -3; // 错误码3：内存分配失败
                        }
                    }
                    break;
                default:
                    break;
            }

            // 读取数据并存储到对应的二维数组中
            if (intData != NULL) {
                for (int i = 0; i < 128; ++i) {
                    for (int j = 0; j < 128; ++j) {
                        if (fscanf(file, "%d", &intData[i][j]) != 1) {
                            // 释放已分配的内存并返回错误
                            for (int k = 0; k < 128; ++k) {
                                free(intData[k]);
                            }
                            free(intData);
                            fclose(file);
                            return -4; // 错误码4：读取数据失败
                        }
                    }
                }
            } else if (floatData != NULL) {
                for (int i = 0; i < 128; ++i) {
                    for (int j = 0; j < 128; ++j) {
                        if (fscanf(file, "%f", &floatData[i][j]) != 1) {
                            // 释放已分配的内存并返回错误
                            for (int k = 0; k < 128; ++k) {
                                free(floatData[k]);
                            }
                            free(floatData);
                            fclose(file);
                            return -4; // 错误码4：读取数据失败
                        }
                    }
                }
            }

            imageIndex++;
        }
    }

    fclose(file);

    #else
        SETERRQ(PETSC_COMM_SELF,MM_UNSUPPORTED_TYPE,"Only support real matrix");
        FunctionReturn(MM_UNSUPPORTED_TYPE);
    #endif

    FunctionReturn(SUCCESS);
}

int PrintMatrixFeatures(MatrixFeatures* mf){
    FunctionBegin;
    printf("rowNum: %d\n", mf->rowNum);
    printf("colNum: %d\n", mf->colNum);
    printf("nnz: %d\n", mf->nnzNum);
    printf("nnzRatio: %lf\n", mf->nnzRatio);
    printf("nnzLower: %lld\n", mf->nnzLower);
    printf("nnzUpper: %lld\n", mf->nnzUpper);
    printf("nnzDiagonal: %lld\n", mf->nnzDiagonal);
    printf("averageNnzEachRow: %lf\n", mf->averageNnzEachRow);
    printf("maxNnzEachRow: %lf\n", mf->maxNnzEachRow);
    printf("minNnzEachRow: %lf\n", mf->minNnzEachRow);
    printf("arrNnzEachRow: %lf\n", mf->arrNnzEachRow);
    printf("maxValue: %lf\n", mf->maxValue);
    printf("maxValueDiagonal: %lf\n", mf->maxValueDiagonal);
    printf("diagonalDominantRatio: %lf\n", mf->diagonalDominantRatio);
    printf("isSymmetric: %d\n", mf->isSymmetric);
    // printf("patternSymm: %lf\n", mf->patternSymm);
    // printf("valueSymm: %lf\n", mf->valueSymm);
    printf("rowVariability: %lf\n", mf->rowVariability);
    printf("colVariability: %lf\n", mf->colVariability);
    printf("trace: %lf\n", mf->trace);
    printf("traceAbs: %lf\n", mf->traceAbs);
    printf("traceASquared: %lf\n", mf->traceASquared);
    printf("norm1: %lf\n", mf->norm1);
    printf("normInf: %lf\n", mf->normInf);
    printf("normF: %lf\n", mf->normF);
    // printf("symmetrySnorm: %lf\n", mf->symmetrySnorm);
    // printf("symmetryAnorm: %lf\n", mf->symmetryAnorm);
    // printf("symmetryFnorm: %lf\n", mf->symmetryFnorm);
    // printf("symmetryFanorm: %lf\n", mf->symmetryFanorm);
    printf("nDummyRows: %d\n", mf->nDummyRows);
    printf("diagZerostat: %d\n", mf->diagZerostat);
    printf("diagDefinite: %d\n", mf->diagDefinite);
    printf("diagonalAverage: %lf\n", mf->diagonalAverage);
    printf("diagonalVariance: %lf\n", mf->diagonalVariance);
    printf("diagonalSign: %d\n", mf->diagonalSign);
    printf("upBand: %d\n", mf->upBand);
    printf("loBand: %d\n", mf->loBand);
    printf("avgDiagDist: %lf\n", mf->avgDiagDist);
    printf("sigmaDiagDist: %lf\n", mf->sigmaDiagDist);
    // printf("relSymm: %lf\n", mf->relSymm);
    printf("kappa: %lf\n", mf->kappa);
    printf("positiveFraction: %lf\n", mf->positiveFraction);
    FunctionReturn(SUCCESS);
}

int DestroyMatrixFeatures(MatrixFeatures* matrixFeatures) {
    FunctionBegin;
    // 检查传入的MatrixFeatures对象是否为空

    int size = 128;
    
    if (matrixFeatures == NULL) {
        return -1; // 错误码，传入参数为空
    }
    // 释放image数组
    if (matrixFeatures->image != NULL) {
        for (int i = 0; i < size; ++i) {
            free(matrixFeatures->image[i]); // 释放每一行
        }
        free(matrixFeatures->image); // 释放指针数组
    }

    // 释放imageNnzDiagonal数组
    if (matrixFeatures->imageNnzDiagonal != NULL) {
        for (int i = 0; i < size; ++i) {
            free(matrixFeatures->imageNnzDiagonal[i]); // 释放每一行
        }
        free(matrixFeatures->imageNnzDiagonal); // 释放指针数组
    }

    // 释放imageRatio数组
    if (matrixFeatures->imageRatio != NULL) {
        for (int i = 0; i < size; ++i) {
            free(matrixFeatures->imageRatio[i]); // 释放每一行
        }
        free(matrixFeatures->imageRatio); // 释放指针数组
    }

    // 释放imageMaxValue数组
    if (matrixFeatures->imageMaxValue != NULL) {
        for (int i = 0; i < size; ++i) {
            free(matrixFeatures->imageMaxValue[i]); // 释放每一行
        }
        free(matrixFeatures->imageMaxValue); // 释放指针数组
    }

    // 释放imageMaxValueDia数组
    if (matrixFeatures->imageMaxValueDia != NULL) {
        for (int i = 0; i < size; ++i) {
            free(matrixFeatures->imageMaxValueDia[i]); // 释放每一行
        }
        free(matrixFeatures->imageMaxValueDia); // 释放指针数组
    }

    // // 释放imageSymmPattern
    // if (matrixFeatures->imageSymmPattern != NULL) {
    //     free(matrixFeatures->imageSymmPattern);
    // }

    // // 释放imageSymmValue
    // if (matrixFeatures->imageSymmValue != NULL) {
    //     free(matrixFeatures->imageSymmValue);
    // }

    // 将MatrixFeatures对象指针设置为NULL，防止野指针出现
    matrixFeatures = NULL;

    FunctionReturn(SUCCESS);
}

int SaveMatrixFeatures(MatrixFeatures* matrixFeatures,char* txtFilePath){
    FunctionBegin;

    FILE* f_write = fopen(txtFilePath, "w");
    if (f_write == NULL) {
        perror("Failed to open file");
        return 1; 
    }

    fprintf(f_write, "rowNum %d\n", matrixFeatures->rowNum);
    fprintf(f_write, "colNum %d\n", matrixFeatures->colNum);
    fprintf(f_write, "nnz %d\n", matrixFeatures->nnzNum);
    fprintf(f_write, "nnzRatio %lf\n", matrixFeatures->nnzRatio);
    fprintf(f_write, "nnzLower %lld\n", matrixFeatures->nnzLower);
    fprintf(f_write, "nnzUpper %lld\n", matrixFeatures->nnzUpper);
    fprintf(f_write, "nnzDiagonal %lld\n", matrixFeatures->nnzDiagonal);
    fprintf(f_write, "averageNnzEachRow %lf\n", matrixFeatures->averageNnzEachRow);
    fprintf(f_write, "maxNnzEachRow %lf\n", matrixFeatures->maxNnzEachRow);
    fprintf(f_write, "minNnzEachRow %lf\n", matrixFeatures->minNnzEachRow);
    fprintf(f_write, "arrNnzEachRow %lf\n", matrixFeatures->arrNnzEachRow);
    fprintf(f_write, "maxValue %lf\n", matrixFeatures->maxValue);
    fprintf(f_write, "maxValueDiagonal %lf\n", matrixFeatures->maxValueDiagonal);
    fprintf(f_write, "diagonalDominantRatio %lf\n", matrixFeatures->diagonalDominantRatio);
    fprintf(f_write, "isSymmetric %d\n", matrixFeatures->isSymmetric);
    // fprintf(f_write, "patternSymm %lf\n", matrixFeatures->patternSymm);
    // fprintf(f_write, "valueSymm %lf\n", matrixFeatures->valueSymm);
    fprintf(f_write, "rowVariability %lf\n", matrixFeatures->rowVariability);
    fprintf(f_write, "colVariability %lf\n", matrixFeatures->colVariability);
    fprintf(f_write, "trace %lf\n", matrixFeatures->trace);
    fprintf(f_write, "traceAbs %lf\n", matrixFeatures->traceAbs);
    fprintf(f_write, "traceASquared %lf\n", matrixFeatures->traceASquared);
    fprintf(f_write, "norm1 %lf\n", matrixFeatures->norm1);      
    fprintf(f_write, "normInf %lf\n", matrixFeatures->normInf);   
    fprintf(f_write, "normF %f\n", matrixFeatures->normF);       
    // fprintf(f_write, "symmetrySnorm %lf\n", matrixFeatures->symmetrySnorm);
    // fprintf(f_write, "symmetryAnorm %lf\n", matrixFeatures->symmetryAnorm);
    // fprintf(f_write, "symmetryFnorm %lf\n", matrixFeatures->symmetryFnorm);
    // fprintf(f_write, "symmetryFanorm %lf\n", matrixFeatures->symmetryFanorm);
    fprintf(f_write, "nDummyRows %d\n", matrixFeatures->nDummyRows);
    fprintf(f_write, "diagZerostat %d\n", matrixFeatures->diagZerostat);
    fprintf(f_write, "diagDefinite %d\n", matrixFeatures->diagDefinite);
    fprintf(f_write, "diagonalAverage %lf\n", matrixFeatures->diagonalAverage);
    fprintf(f_write, "diagonalVariance %lf\n", matrixFeatures->diagonalVariance);
    fprintf(f_write, "diagonalSign %d\n", matrixFeatures->diagonalSign);
    fprintf(f_write, "upBand %d\n", matrixFeatures->upBand);
    fprintf(f_write, "loBand %d\n", matrixFeatures->loBand);
    fprintf(f_write, "avgDiagDist %lf\n", matrixFeatures->avgDiagDist);
    fprintf(f_write, "sigmaDiagDist %lf\n", matrixFeatures->sigmaDiagDist);  
    // fprintf(f_write, "relSymm %lf\n", matrixFeatures->relSymm); 
    fprintf(f_write, "kappa %lf\n", matrixFeatures->kappa); 
    fprintf(f_write, "positiveFraction %lf\n", matrixFeatures->positiveFraction); 

    // 写入图片数据
    int size = 128; // 图片大小为128x128
    fprintf(f_write, "image\n");
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            fprintf(f_write, "%d ", matrixFeatures->image[i][j]);
        }
        fprintf(f_write, "\n");
    }

    fprintf(f_write, "imageNnzDiagonal\n");
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            fprintf(f_write, "%d ", matrixFeatures->imageNnzDiagonal[i][j]);
        }
        fprintf(f_write, "\n");
    }

    fprintf(f_write, "imageRatio\n");
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            fprintf(f_write, "%f ", matrixFeatures->imageRatio[i][j]);
        }
        fprintf(f_write, "\n");
    }

    fprintf(f_write, "imageMaxValue\n");
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            fprintf(f_write, "%f ", matrixFeatures->imageMaxValue[i][j]);
        }
        fprintf(f_write, "\n");
    }

    fprintf(f_write, "imageMaxValueDia\n");
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            fprintf(f_write, "%f ", matrixFeatures->imageMaxValueDia[i][j]);
        }
        fprintf(f_write, "\n");
    }

    fclose(f_write);

    FunctionReturn(SUCCESS);
}