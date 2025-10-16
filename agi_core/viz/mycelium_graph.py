def export_edges_csv(net,path:str):
    with open(path,'w',encoding='utf-8') as f:
        f.write('src,dst,weight\n')
        for u,nbrs in net.edges.items():
            for v,w in nbrs.items(): f.write(f"{u},{v},{w:.6f}\n")
