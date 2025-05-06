# 设置编译器和编译选项
CC      = gcc
CFLAGS  = -g -O3 -fopenmp -std=c99 -pthread

# ========== 配置 PETSc 和 SLEPc ==========

# 允许通过环境变量传入 PETSC_DIR 和 PETSC_ARCH，如果没有就用默认值
PETSC_DIR = /data1/home/wujunxiang/aisolver/src/petsc-3.20.0
PETSC_ARCH = arch-linux-c-debug

# 检查 petscvariables 文件是否存在
ifeq (,$(wildcard $(PETSC_DIR)/$(PETSC_ARCH)/lib/petsc/conf/petscvariables))
$(error PETSc配置错误：$(PETSC_DIR)/$(PETSC_ARCH)/lib/petsc/conf/petscvariables 不存在，请正确设置PETSC_DIR和PETSC_ARCH)
endif

# 加载 PETSc 变量
include $(PETSC_DIR)/$(PETSC_ARCH)/lib/petsc/conf/petscvariables

# 包含头文件路径
INCLUDES = -I$(PETSC_DIR)/include $(PETSC_CC_INCLUDES)

# 链接库
LDFLAGS  = $(PETSC_LIB) $(SLEPC_LIB) -lm

# ========== 项目文件配置 ==========

# 最终生成的可执行文件名
TARGET = main

# 需要编译的源文件列表
SRC_FILES = main.c AIFeatures.c mmio.c

# 自动生成对应的 .o 目标文件
OBJ_FILES = $(SRC_FILES:.c=.o)

# ========== 编译规则 ==========

# 默认目标：生成可执行文件
$(TARGET): $(OBJ_FILES)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# 通用编译规则：.c -> .o
%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

# ========== 清理命令 ==========

# 执行 make clean 删除中间文件和可执行文件
clean:
	rm -f *.o $(TARGET)
