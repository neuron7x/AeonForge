import csv

def write_trajectory_csv(positions,out_csv:str):
    with open(out_csv,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['t','x','y'])
        for t,(x,y) in enumerate(positions): w.writerow([t,x,y])
