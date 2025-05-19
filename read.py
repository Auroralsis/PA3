import sys

def read_csr_data(file_path):
    try:
        with open(file_path, 'r') as file:
            ptr_line = file.readline().strip()
            idx_line = file.readline().strip()  # 读取但不使用
            val_line = file.readline().strip()  # 读取但不使用

        # 将行指针转换为整数列表
        row_ptr = list(map(int, ptr_line.split()))

        if len(row_ptr) < 2:
            raise ValueError("Invalid CSR format: insufficient row pointer information.")

        # 计算每一行的非零元素数量
        row_counts = [row_ptr[i + 1] - row_ptr[i] for i in range(len(row_ptr) - 1)]

        return row_counts

    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python read.py <relative_file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    row_counts = read_csr_data(file_path)

    if row_counts is not None:
        print("Number of non-zero elements per row:")
        for i, count in enumerate(row_counts):
            print(f"Row {i}: {count} non-zero elements.")

if __name__ == "__main__":
    main()