def build_gif(png_paths,out_path,fps:int=6):
    try:
        import imageio
        imgs=[imageio.v2.imread(p) for p in png_paths]
        imageio.mimsave(out_path, imgs, duration=max(1e-3, 1.0/float(fps)))
        return True
    except Exception:
        return False
