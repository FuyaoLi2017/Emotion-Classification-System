def display(img):
    print("shape is: ", img.shape)
    plt.imshow(img)
    plt.show()
    
def extract(name, paths):
    feat_list = []
    norm_feat_list = []
    i = 0
    for p in paths:
        i += 1
        if i % 1000 == 0:
            print(str(i) + " has been extracted")
        [feat, norm_feat] = model.extract_feat(p)
        feat_list.append(np.squeeze(feat))
        norm_feat_list.append(norm_feat)
        
    with open(name + "_feat", "wb") as f:
        pickle.dump(feat_list, f)
    with open(name + "_norm", "wb") as f:
        pickle.dump(norm_feat_list, f)
    print("done: " + name + " feature saved to disk")