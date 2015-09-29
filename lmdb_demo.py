import cv2 as cv
import lmdb
import caffe
import lmdb_util as lu

if __name__ == '__main__':
    # set lmdb path
    img_lmdb_path = './image_train.lmdb'
    bbox_lmdb_path = './bbox_train.lmdb'

    # Initialize the lmdbs
    lu.del_and_create(img_lmdb_path)
    lu.del_and_create(bbox_lmdb_path)

    img_env, img_txn, bbox_env, bbox_txn = None, None, None, None

    # Image lmdb configuration
    img_env = lmdb.Environment(img_lmdb_path, map_size=1099511627776)
    img_txn = img_env.begin(write=True, buffers=True)

    # Bounding box lmdb configuration
    bbox_env = lmdb.Environment(bbox_lmdb_path, map_size=1099511627776)
    bbox_txn = bbox_env.begin(write=True, buffers=True)

    # Initialize random keys
    keys = np.arange(len(list))
    np.random.shuffle(keys)

    for image_file_path in images:

        # Read data from file
        img = cv.imread(img_file_path, cv.IMREAD_COLOR)
        bbox = None

        # Push to lmdb
        train_img_datum = lu.generate_img_datum(img, label=attri_label)
        bbox_datum = lu.generate_array_datum(ext_bbox)

        # Generate a key and push to buffer
        key = '%010d' % keys[count]
        img_txn.put(key, train_img_datum.SerializeToString())
        bbox_txn.put(key, bbox_datum.SerializeToString())

        if count % 10000 == 0:
            # Write to lmdb every 10000 times
            img_txn.commit()
            bbox_txn.commit()
            img_txn = img_env.begin(write=True, buffers=True)
            bbox_txn = bbox_env.begin(write=True, buffers=True)

# Close the lmdb
img_txn.commit()
bbox_txn.commit()
img_env.close()
bbox_env.close()
