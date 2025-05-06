/*
 * @Author: hnu_hss
 * @Date: 2024-11-28 16:01:44
 * @LastEditTime: 2024-12-25 11:41:42
 * @FilePath: \aisolver\include\AIFeatures.h
 * @Description: 
 * 
 * Copyright (c) 2024 by hnu_hss, All Rights Reserved. 
 */

#pragma once

#include <stdbool.h>
#include "Err.h"


typedef struct MatrixFeatures{
    int rowNum;                  // 矩阵的行数
    int colNum;                  // 矩阵的列数   
    long long nnz;               // 矩阵中非零元素的数量
    double nnzRatio;             // 非零元素与总元素的比例
    long long nnzLower;          // 矩阵下三角部分的非零元素数量
    long long nnzUpper;          // 矩阵上三角部分的非零元素数量
    long long nnzDiagonal;       // 矩阵对角线上的非零元素数量
    double averageNnzEachRow;    // 每行平均非零元素的数量
    double maxNnzEachRow;        // 每行非零元素的最大数量
    double minNnzEachRow;        // 每行非零元素的最小数量
    double arrNnzEachRow;        // 每行非零元素的标准差
    double maxValue;             // 矩阵中的最大值
    double maxValueDiagonal;     // 对角线上元素的最大值
    double diagonalDominantRatio;
    double diagonalDominantRatio_1; // 对角占优比率
    double diagonalDominantRatio_2; // 对角占优比率
    double diagonalDominantRatio_3; // 对角占优比率
    int isSymmetric;             // 矩阵是否对称
    // double patternSymm;          // 模式对称性
    // double valueSymm;            // 值对称性
    double rowVariability;       // 行的变异性
    double colVariability;       // 列的变异性
    double trace;                // 矩阵的迹
    double traceAbs;             // 矩阵迹的绝对值
    double traceASquared;        // 矩阵迹的平方
    double norm1;                // 矩阵的1-范数
    double normInf;              // 矩阵的无穷范数
    double normF;                // 矩阵的Frobenius范数
    // double symmetrySnorm;        // 对称性结构范数
    // double symmetryAnorm;        // 对称性绝对范数
    // double symmetryFnorm;        // 对称性Frobenius范数
    // double symmetryFanorm;       // 对称性绝对Frobenius范数
    int nDummyRows;              // 虚拟行的数量
    int diagZerostat;            // 对角线零元素状态
    int diagDefinite;            // 对角线是否正定
    double diagonalAverage;      // 对角线元素的平均值
    double diagonalVariance;     // 对角线元素的方差
    int diagonalSign;            // 对角线元素的符号数量
    int upBand;                  // 上带宽度
    int loBand;                  // 下带宽度
    double avgDiagDist;          // 对角线元素的平均距离
    double sigmaDiagDist;        // 对角线元素的标准差
    // double relSymm;              // 相对对称性
    double kappa;                // Kappa统计量
    double positiveFraction;     // 正元素的比例

    int nnzMtx;
    int nnzNum;

    int** image;
    int** imageNnzDiagonal;
    float** imageRatio;
    float** imageMaxValue;
    float** imageMaxValueDia;
    // float* imageSymmPattern; //len = 128x128
    // float* imageSymmValue; //len = 128x128
}MatrixFeatures;

typedef struct VectorFeatures{
    // TODO
}VectorFeatures;


/*--------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/


/**
 * @description: 从mtx文件提取矩阵特征，创建Matrix类型矩阵，支持对称/非对称，目前仅支持实数。
 * @param {MatrixFeatures*} matrixFeatures 该矩阵对应的MatrixFeatures类型矩阵特征
 * @param {char*} mtxFilePath mtx文件路径
 * @return {ErrorCode} 错误码
 */
ErrorCode CreateMatrixFeaturesFromMTX(MatrixFeatures* matrixFeatures, char* mtxFilePath);

/**
 * @description: 根据输入矩阵的CSR格式，提取矩阵特征，创建Matrix类型矩阵，支持对称/非对称，目前仅支持实数。
 * @param {MatrixFeatures*} matrixFeatures 该矩阵对应的MatrixFeatures类型矩阵特征
 * @param {int} nRow 输入矩阵的行数
 * @param {int} nCol 输入矩阵的列数
 * @param {bool} matSymType 输入矩阵的对称性，false表示非对称，true表示对称
 * @param {bool} matValueType 输入矩阵的数值类型，false表示实数，true表示复数
 * @param {int*} rowOffset 输入矩阵CSR格式的rowOffset数组
 * @param {int*} colIndex 输入矩阵CSR格式的colIndex数组
 * @param {double*} value 输入矩阵CSR格式的value数组
 * @return {ErrorCode} 错误码
 */
ErrorCode CreateMatrixFeaturesFromCSR(MatrixFeatures* matrixFeatures, int nRow, int nCol, bool matSymType, bool matValueType, int* rowOffset, int* colIndex, double* value);

/**
 * @description: 根据输入矩阵的COO格式，提取矩阵特征，创建Matrix类型矩阵，支持对称/非对称，目前仅支持实数。
 * @param {MatrixFeatures*} matrixFeatures 该矩阵对应的MatrixFeatures类型矩阵特征
 * @param {int} nRow 输入矩阵的行数
 * @param {int} nCol 输入矩阵的列数
 * @param {int} nnz 输入矩阵的非零元个数
 * @param {bool} matSymType 输入矩阵的对称性，false表示非对称，true表示对称
 * @param {bool} matValueType 输入矩阵的数值类型，false表示实数，true表示复数
 * @param {int*} rowId 输入矩阵COO格式的rowId数组
 * @param {int*} colId 输入矩阵COO格式的colId数组
 * @param {double*} value 输入矩阵COO格式的value数组
 * @return {ErrorCode} 错误码
 */
ErrorCode CreateMatrixFeaturesFromCOO(MatrixFeatures* matrixFeatures, int nRow, int nCol, int nnz, bool matSymType, bool matValueType, int* rowId, int* colId, double* value);

/**
 * @description: 直接从矩阵特征文件中读入矩阵特征，创建Matrix类型矩阵，目前仅支持实数。
 * @param {MatrixFeatures*} matrixFeatures 读入的MatrixFeatures类型矩阵特征
 * @param {char*} featureFilePath 矩阵特征文件路径
 * @return {ErrorCode} 错误码
 */
ErrorCode CreateMatrixFeaturesFromTXT(MatrixFeatures* matrixFeatures, char* featureFilePath);

int SaveMatrixFeatures(MatrixFeatures* matrixFeatures,char* txtFilePath);
int writeImage(MatrixFeatures* mtx,const char* outputpath);

/**
 * @description: 打印矩阵特征
 * @param {MatrixFeatures*} matrixFeatures MatrixFeatures对象
 * @return {ErrorCode} 错误码
 */
ErrorCode PrintMatrixFeatures(MatrixFeatures* matrixFeatures);

/**
 * @description: 销毁MatrixFeatures对象
 * @param {MatrixFeatures*} matrixFeatures 待销毁的MatrixFeatures对象
 * @return {ErrorCode} 错误码
 */
ErrorCode DestroyMatrixFeatures(MatrixFeatures* matrixFeatures);


/*--------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/


/**
 * @description: 根据输入向量，提取向量特征，创建VectorFeatures类型向量特征，目前仅支持实数。
 * @param {VectorFeatures*} vectorFeatures 该向量对应的VectorFeatures类型向量特征
 * @param {int} nRow 输入向量的行数
 * @param {bool} vecVauleType 输入向量的数值类型，false表示实数，true表示复数
 * @param {double*} value 输入向量的value数组
 * @return {ErrorCode} 错误码
 */
ErrorCode CreateVectorFeaturesFromDense(VectorFeatures* vectorFeatures, int nRow, bool vecVauleType, double* value);

/**
 * @description: 从rhs文件读入向量，提取向量特征，创建VectorFeatures类型向量特征，目前仅支持实数。
 * @param {VectorFeatures*} vectorFeatures 该向量对应的VectorFeatures类型向量特征
 * @param {char*} rhsFilePath rhs文件路径
 * @return {ErrorCode} 错误码
 */
ErrorCode CreateVectorFeaturesFromRHS(VectorFeatures* vectorFeatures, char* rhsFilePath);

/**
 * @description: 从mtx文件读入向量，提取向量特征，创建VectorFeatures类型向量特征，目前仅支持实数。
 * @param {VectorFeatures*} vectorFeatures 该向量对应的VectorFeatures类型向量特征
 * @param {char*} mtxFilePath mtx文件路径
 * @return {ErrorCode} 错误码
 */
ErrorCode CreateVectorFeaturesFromMTX(VectorFeatures* vectorFeatures, char* mtxFilePath);

/**
 * @description: 直接从向量特征文件读入向量特征，创建VectorFeatures类型向量特征，目前仅支持实数。
 * @param {VectorFeatures*} vectorFeatures 读入的VectorFeatures类型向量特征
 * @param {char*} txtFilePath 向量特征文件路径
 * @return {ErrorCode} 错误码
 */
ErrorCode CreateVectorFeaturesFromTXT(VectorFeatures* vectorFeatures, char* txtFilePath);

/**
 * @description: 销毁VectorFeatures对象
 * @param {VectorFeatures*} vectorFeatures 待销毁的VectorFeatures对象
 * @return {ErrorCode} 错误码
 */
ErrorCode DestroyVectorFeatures(VectorFeatures* vectorFeatures);