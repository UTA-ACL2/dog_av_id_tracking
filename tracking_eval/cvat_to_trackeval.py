from pathlib import Path
import xml.etree.ElementTree as ET


ANNOTATIONS_DIR = Path("annotations")
OUT_DIR = Path("MOT_eval/gt")


def convert_xml(xml_path, output_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    count = 0

    with open(output_path, "w") as f:

        for track in root.findall("track"):

            label = track.attrib.get("label", "").lower()

            # only keep dogs
            if label != "dog" and label != "Dog":
                continue

            track_id = int(track.attrib["id"])

            for box in track.findall("box"):

                # skip boxes where object is outside
                if box.attrib.get("outside", "0") == "1":
                    continue

                # CVAT frames are 0-indexed
                # MOT format is 1-indexed
                frame = int(box.attrib["frame"]) + 1

                xtl = float(box.attrib["xtl"])
                ytl = float(box.attrib["ytl"])
                xbr = float(box.attrib["xbr"])
                ybr = float(box.attrib["ybr"])

                width = xbr - xtl
                height = ybr - ytl

                # MOTChallenge format:
                # frame,id,x,y,w,h,conf,class,visibility
                f.write(
                    f"{frame},{track_id},"
                    f"{xtl},{ytl},"
                    f"{width},{height},"
                    "1,1,1\n"
                )

                count += 1


    print(
        f"{xml_path.name}: {count} boxes -> {output_path}"
    )

    return count



def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    xml_files = sorted(
        ANNOTATIONS_DIR.glob("*.xml"),
        key=lambda x: int(x.stem)
    )


    for xml_path in xml_files:

        video_id = xml_path.stem


        output_path = OUT_DIR / f"{video_id}.txt"


        count = convert_xml(
            xml_path,
            output_path
        )


        if count == 0:
            print(
                f"WARNING: no dog annotations found in {xml_path.name}"
            )


if __name__ == "__main__":
    main()