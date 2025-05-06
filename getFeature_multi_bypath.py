import os 
from multiprocessing import Pool

def parallel_run_uniprocess_solver(mtx_name_arr, core):
    for mtx_name in mtx_name_arr:
        feature_output_path = "./feature/"
        
        # 提取文件名（不含扩展名）
        file_ends = "." + mtx_name.split(".")[-1]
        name = mtx_name.split("/")[-1].replace(file_ends, "")
        
        # 输入路径处理
        inputpath = mtx_name.strip()
        
        # 构造特征文件输出路径
        feature_output_path += name + ".features"

        # 检查特征文件是否已存在
        if os.path.exists(feature_output_path):
            process_line = f"rank {core} is skipping {mtx_name} as {feature_output_path} exists."
            cmd = f"./main {inputpath} {feature_output_path}"
            with open("./log_wjx.txt", "a+") as f:
                f.write(process_line + "\n")
                f.write(f"Skipped command: {cmd}\n")
            print(process_line)
            print(f"Skipped command: {cmd}")
            continue

        # 执行处理逻辑
        process_line = f"rank {core} is dealing with {mtx_name}"
        with open("./log.txt", "a+") as f:
            f.write(process_line + "\n")
        
        print(process_line)
        cmd = f"./main {inputpath} {feature_output_path}"
        print(cmd)
        os.system(cmd)

def data_partition(num):
    names = []
    path = "/data2/csu_structure_matrix/csu_structure_matrix/"
    
    # 获取所有.mtx文件路径
    for file in os.listdir(path):
        if file.endswith(".mtx"):
            full_path = os.path.join(path, file)
            names.append(full_path)
    
    # 按进程数分片
    file_map = {}
    for i in range(num):
        file_map[i] = [names[j] for j in range(len(names)) if j % num == i]
    return file_map

def run_parallel_uni_solver(process_num):
    file_map = data_partition(process_num)
    p = Pool(process_num)
    for i in range(process_num):
        p.apply_async(parallel_run_uniprocess_solver, args=(file_map[i], i))
    print("Waiting for all subprocesses done...")
    p.close()
    p.join()
    print("All subprocess done...")

if __name__ == "__main__":
    run_parallel_uni_solver(10)
    print("Finish running!")