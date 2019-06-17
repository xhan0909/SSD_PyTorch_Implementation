import os
import re
import sys
import json
import xml.etree.ElementTree as ET

data = dict()
data['images'] = []
data['type'] = 'instances'
data['annotations'] = []
data['categories'] = []

category_set = dict()
image_set = set()

category_item_id = 0
image_id = 0
annotation_id = 0

def set_img_id(fp, imgId):
    global image_id, file_path
    file_path = fp
    image_id=imgId


def addCatItem(name):
    global category_item_id
    category_item = dict()
    category_item['supercategory'] = 'none'
    category_item_id += 1
    category_item['id'] = category_item_id
    category_item['name'] = name
    data['categories'].append(category_item)
    category_set[name] = category_item_id
    return category_item_id


def addImgItem(file_name, size, image_id):
    if file_name is None:
        raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
        raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
        raise Exception('Could not find height tag in xml file.')
    
    image_item = dict()
    image_item['id'] = image_id
    image_item['file_name'] = file_path
    image_item['width'] = size['width']
    image_item['height'] = size['height']
    data['images'].append(image_item)
    image_set.add(file_name)
    return image_id


def addAnnoItem(object_name, image_id, category_id, bbox):
    global annotation_id
    annotation_item = dict()
    annotation_item['segmentation'] = []
    seg = []
    #bbox[] is x,y,w,h
    #left_top
    seg.append(bbox[0])
    seg.append(bbox[1])
    #left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    #right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    #right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])

    annotation_item['segmentation'].append(seg)

    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    annotation_id += 1
    annotation_item['id'] = annotation_id
    data['annotations'].append(annotation_item)


def convert_xml_path(img_path):
    """
    Convert image path to its corresponding xml file path.

    :param img_path: text file containing paths of image files
    :return: a list of xml file path
    """
    paths = []
    with open(img_path, 'r') as f:
        while True:
            path = f.readline()
            if not path: break
            path = re.sub(r'.jp(\w+)', '.xml', path.strip("'\n"))
            path = path.replace('Image Training Data/', 'xml/')
            path = path.replace('Microfibers/', '')
            paths.append(path)

    return paths


def parseXmlFiles(xml_path): 
    for f in xml_path:
        if not f.endswith('.xml'):
            continue
        
        bndbox = dict()
        size = dict()
        current_image_id = None
        current_category_id = None
        file_name = None
        size['width'] = None
        size['height'] = None
        size['depth'] = None

        xml_file = f
        print(xml_file)

        #get substring between / and . and set global variable image id
        pattern=r'(?<=\/)(.*)(?=\.xml)'
        imgId= re.search(pattern, xml_file).group(0)
        imgId = imgId.split('/')[-1]
        print(imgId)
        im_path = xml_file.replace('xml/', 'Image Training Data/')
        im_path = im_path.split('/')
        tmp_path = im_path.copy()
        im_path = tmp_path[:-1] + ['Microfibers'] + [tmp_path[-1].replace('.xml', '.jpeg')]
        set_img_id('/'.join(im_path), imgId)



        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.tag != 'annotation':
            raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

        #elem is <folder>, <filename>, <size>, <object>
        for elem in root:
            current_parent = elem.tag
            current_sub = None
            object_name = None
            
            if elem.tag == 'folder':
                continue
            
            if elem.tag == 'filename':
                file_name = elem.text
                if file_name in category_set:
                    raise Exception('file_name duplicated')
                
            #add img item only after parse <size> tag
            elif current_image_id is None and file_name is not None and size['width'] is not None:
                if file_name not in image_set:
                    current_image_id = addImgItem(file_name, size,image_id)
                    print('add image with {} and {}'.format(file_name, size))
                else:
                    raise Exception('duplicated image: {}'.format(file_name)) 
            #subelem is <width>, <height>, <depth>, <name>, <bndbox>
            for subelem in elem:
                bndbox ['xmin'] = None
                bndbox ['xmax'] = None
                bndbox ['ymin'] = None
                bndbox ['ymax'] = None
                
                current_sub = subelem.tag
                if current_parent == 'object' and subelem.tag == 'name':
                    object_name = subelem.text
                    if object_name not in category_set:
                        current_category_id = addCatItem(object_name)
                    else:
                        current_category_id = category_set[object_name]

                elif current_parent == 'size':
                    if size[subelem.tag] is not None:
                        raise Exception('xml structure broken at size tag.')
                    size[subelem.tag] = int(subelem.text)

                #option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
                for option in subelem:
                    if current_sub == 'bndbox':
                        if bndbox[option.tag] is not None:
                            raise Exception('xml structure corrupted at bndbox tag.')
                        bndbox[option.tag] = int(option.text)

                #only after parse the <object> tag
                if bndbox['xmin'] is not None:
                    if object_name is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_image_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_category_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    bbox = []
                    #x
                    bbox.append(bndbox['xmin'])
                    #y
                    bbox.append(bndbox['ymin'])
                    #w
                    bbox.append(bndbox['xmax'] - bndbox['xmin'])
                    #h
                    bbox.append(bndbox['ymax'] - bndbox['ymin'])
                    print('add annotation with {},{},{},{}'.format(object_name, current_image_id, current_category_id, bbox))
                    addAnnoItem(object_name, current_image_id, current_category_id, bbox )


if __name__ == '__main__':
    img_path = sys.argv[1]  # a txt file contains paths of all the data (image)
    xml_path = convert_xml_path(img_path)
    json_file = sys.argv[2]
    parseXmlFiles(xml_path)
    json.dump(data, open(json_file, 'w'))
