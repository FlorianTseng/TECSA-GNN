import os
import re
import csv

cif_dir = r"E:\paper1-2025\TECSAGNN\new-mater\pred\cif_files"
gap_file = r"E:\paper1-2025\TECSAGNN\new-mater\cgcnn-gap\band-gap.csv"
output_file = "prop.csv"

gap_data = {}
with open(gap_file, 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) < 2:
            parts = line.strip().split()
        if len(parts) >= 2:
            mpid, gap = parts[0], parts[1]
            gap_data[mpid] = gap

temperatures = [str(temp) for temp in range(100, 1400, 100)]

output_rows = []

for cif_file in os.listdir(cif_dir):
    if cif_file.endswith('.cif') and cif_file.startswith('mp-'):
        mpid = cif_file.split('.')[0]
        
        formula = ""
        cif_path = os.path.join(cif_dir, cif_file)
        with open(cif_path, 'r') as f:
            for line in f:
                if line.startswith('_chemical_formula_sum'):
                    match = re.search(r"'([^']+)'", line)
                    if match:
                        formula_parts = match.group(1).replace(' ', '').split()
                        elements = []
                        for part in formula_parts:
                            element = re.sub(r'\d+', '', part)
                            elements.append(element)
                        formula = ''.join(elements)
                    break
        
        gap = gap_data.get(mpid, "0")
        if gap == "0":
            print(f"警告: {mpid} 的带隙值未找到，使用默认值0")
        
        for dopant_type in ['p', 'n']:
            for doping in [0.0001, 0.001, 0.01, 0.1, 1]:
                temp_values = ['1'] * len(temperatures)
                
                row = [
                    formula, mpid, 
                    *temp_values, 
                    dopant_type, doping, gap
                ]
                
                output_rows.append(row)

with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['formula', 'mpid'] + temperatures + ['type', 'dope', 'gap'])
    writer.writerows(output_rows)

print(f"已成功生成 {output_file} 文件，包含 {len(output_rows)} 行数据")
print(f"带隙数据共加载了 {len(gap_data)} 个MPID")