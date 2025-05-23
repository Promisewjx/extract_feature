/*
 * @Author: hnu_hss
 * @Date: 2024-12-11 20:08:37
 * @LastEditTime: 2024-12-24 12:09:59
 * @Description: 
 * 
 * Copyright (c) 2024 by hnu_hss, All Rights Reserved. 
 */
#pragma once

#include "mmio.h"
#include "petscsystypes.h"
#include "petscsys.h"
#include "petscerror.h"

typedef int ErrorCode;
#define SUCCESS 0

 /**
 * @description: 根据输入的错误码，输出对应的错误信息。若无错误，则不输出。
 * @param {ErrorCode} errorCode 错误码
 */
// void CheckErrorCode(ErrorCode errorCode);

#define CheckErrorCode PetscCall
#define FunctionBegin PetscFunctionBegin
#define FunctionReturn PetscFunctionReturn


/*-------------mmio错误码-------------*/
#define MM_COULD_NOT_READ_FILE  11
#define MM_PREMATURE_EOF        12
#define MM_NOT_MTX              13
#define MM_NO_HEADER            14
#define MM_UNSUPPORTED_TYPE     15
#define MM_LINE_TOO_LONG        16
#define MM_COULD_NOT_WRITE_FILE 17
// 下面为添加的
#define MM_COULD_NOT_OPEN_FILE  18
#define MM_COULD_NOT_READ_BANNER 19
#define MM_COULD_NOT_READ_MTX_CRD_SIZE 20
#define MM_COULD_NOT_ALLOC_MEM 21
#define MM_COULD_NOT_READ_CRD_VALUE 22



/*---------------------------------------------PETSC错误码--------------------------------------------------*/
// typedef int PETSC_ERROR_CODE_TYPEDEF;
// PETSC_ERROR_CODE_TYPEDEF enum PETSC_ERROR_CODE_NODISCARD {
//   PETSC_SUCCESS                   = 0,
//   PETSC_ERR_BOOLEAN_MACRO_FAILURE = 1, /* do not use */

//   PETSC_ERR_MIN_VALUE = 54, /* should always be one less than the smallest value */

//   PETSC_ERR_MEM            = 55, /* unable to allocate requested memory */
//   PETSC_ERR_SUP            = 56, /* no support for requested operation */
//   PETSC_ERR_SUP_SYS        = 57, /* no support for requested operation on this computer system */
//   PETSC_ERR_ORDER          = 58, /* operation done in wrong order */
//   PETSC_ERR_SIG            = 59, /* signal received */
//   PETSC_ERR_FP             = 72, /* floating point exception */
//   PETSC_ERR_COR            = 74, /* corrupted PETSc object */
//   PETSC_ERR_LIB            = 76, /* error in library called by PETSc */
//   PETSC_ERR_PLIB           = 77, /* PETSc library generated inconsistent data */
//   PETSC_ERR_MEMC           = 78, /* memory corruption */
//   PETSC_ERR_CONV_FAILED    = 82, /* iterative method (KSP or SNES) failed */
//   PETSC_ERR_USER           = 83, /* user has not provided needed function */
//   PETSC_ERR_SYS            = 88, /* error in system call */
//   PETSC_ERR_POINTER        = 70, /* pointer does not point to valid address */
//   PETSC_ERR_MPI_LIB_INCOMP = 87, /* MPI library at runtime is not compatible with MPI user compiled with */

//   PETSC_ERR_ARG_SIZ          = 60, /* nonconforming object sizes used in operation */
//   PETSC_ERR_ARG_IDN          = 61, /* two arguments not allowed to be the same */
//   PETSC_ERR_ARG_WRONG        = 62, /* wrong argument (but object probably ok) */
//   PETSC_ERR_ARG_CORRUPT      = 64, /* null or corrupted PETSc object as argument */
//   PETSC_ERR_ARG_OUTOFRANGE   = 63, /* input argument, out of range */
//   PETSC_ERR_ARG_BADPTR       = 68, /* invalid pointer argument */
//   PETSC_ERR_ARG_NOTSAMETYPE  = 69, /* two args must be same object type */
//   PETSC_ERR_ARG_NOTSAMECOMM  = 80, /* two args must be same communicators */
//   PETSC_ERR_ARG_WRONGSTATE   = 73, /* object in argument is in wrong state, e.g. unassembled mat */
//   PETSC_ERR_ARG_TYPENOTSET   = 89, /* the type of the object has not yet been set */
//   PETSC_ERR_ARG_INCOMP       = 75, /* two arguments are incompatible */
//   PETSC_ERR_ARG_NULL         = 85, /* argument is null that should not be */
//   PETSC_ERR_ARG_UNKNOWN_TYPE = 86, /* type name doesn't match any registered type */

//   PETSC_ERR_FILE_OPEN       = 65, /* unable to open file */
//   PETSC_ERR_FILE_READ       = 66, /* unable to read from file */
//   PETSC_ERR_FILE_WRITE      = 67, /* unable to write to file */
//   PETSC_ERR_FILE_UNEXPECTED = 79, /* unexpected data in file */

//   PETSC_ERR_MAT_LU_ZRPVT = 71, /* detected a zero pivot during LU factorization */
//   PETSC_ERR_MAT_CH_ZRPVT = 81, /* detected a zero pivot during Cholesky factorization */

//   PETSC_ERR_INT_OVERFLOW   = 84,
//   PETSC_ERR_FLOP_COUNT     = 90,
//   PETSC_ERR_NOT_CONVERGED  = 91,  /* solver did not converge */
//   PETSC_ERR_MISSING_FACTOR = 92,  /* MatGetFactor() failed */
//   PETSC_ERR_OPT_OVERWRITE  = 93,  /* attempted to over write options which should not be changed */
//   PETSC_ERR_WRONG_MPI_SIZE = 94,  /* example/application run with number of MPI ranks it does not support */
//   PETSC_ERR_USER_INPUT     = 95,  /* missing or incorrect user input */
//   PETSC_ERR_GPU_RESOURCE   = 96,  /* unable to load a GPU resource, for example cuBLAS */
//   PETSC_ERR_GPU            = 97,  /* An error from a GPU call, this may be due to lack of resources on the GPU or a true error in the call */
//   PETSC_ERR_MPI            = 98,  /* general MPI error */
//   PETSC_ERR_RETURN         = 99,  /* PetscError() incorrectly returned an error code of 0 */
//   PETSC_ERR_MEM_LEAK       = 100, /* memory alloc/free imbalance */
//   PETSC_ERR_MAX_VALUE      = 101, /* this is always the one more than the largest error code */

//   /*
//     do not use, exist purely to make the enum bounds equal that of a regular int (so conversion
//     to int in main() is not undefined behavior)
//   */
//   PETSC_ERR_MIN_SIGNED_BOUND_DO_NOT_USE = INT_MIN,
//   PETSC_ERR_MAX_SIGNED_BOUND_DO_NOT_USE = INT_MAX
// } PETSC_ERROR_CODE_ENUM_NAME;