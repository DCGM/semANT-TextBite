import numpy as np
import lxml.etree as ET
from collections import defaultdict

from textbite.geometry import polygon_centroid, get_lines_polygon
from pero_ocr.document_ocr.layout import RegionLayout


class UnsupportedLayoutError(Exception):
    pass


class PageXMLEnhancer:
    def __init__(self):
        self.y_margin = 5  # if two regions in a bite are not at least this many y-pixels aside, consider them overlapping

    def process(self, layout, bites):
        self.split_regions(layout, bites)

        layout_bites = [[line.id for line in r.lines] for r in layout.regions]
        all_lines_ids = list(line.id for line in layout.lines_iterator())
        region_centers = [polygon_centroid(*zip(*r.polygon)) for r in layout.regions]

        assert len(set(sum((b.lines for b in bites), []))) == sum(len(b.lines) for b in bites), 'Bites have to be disjunct'
        assert set(sum((b.lines for b in bites), [])).issubset(all_lines_ids), 'Bites must only have lines from the layout'

        coverage = [self.get_covering_bites(bite.lines, layout_bites) for bite in bites]

        for bite in coverage:
            for r in bite:
                assert r[1] is True, 'Unpure regions cannot occur at this point'

        reading_order = []
        for bite in coverage:
            reading_order.append(self.get_bite_reading_order(region_centers, [r[0] for r in bite]))

        reading_order_root = self.get_reading_order_xml(reading_order, layout)

        out_xml = ET.fromstring(layout.to_pagexml_string())
        ns = out_xml.nsmap[None]
        page_qname = ET.QName(ns, 'Page')
        out_xml.find(page_qname).insert(0, reading_order_root)

        ET.indent(out_xml)
        return ET.tostring(out_xml, pretty_print=True, encoding=str)

    def split_regions(self, layout, bites):
        new_regions = []
        for region in layout.regions:
            covering_bites = defaultdict(list)
            for line in region.lines:
                for bite_id, bite in enumerate(bites):
                    if line.id in bite.lines:
                        covering_bites[bite_id].append(line.id)
                        break
                else:
                    covering_bites[-1].append(line.id)

            #  regions covered by a single bite need no further attention
            if len(covering_bites) == 1:
                new_regions.append(region)
            else:
                for bite, lines in covering_bites.items():
                    bite_lines = [line for line in region.lines if line.id in lines]
                    polygon = get_lines_polygon(bite_lines)
                    new_region = RegionLayout(f'{region.id}_{bite}', polygon)
                    new_region.lines = bite_lines
                    new_regions.append(new_region)

        layout.regions = new_regions

    def get_covering_bites(self, lines_to_cover, bites):
        lines_to_cover = set(lines_to_cover)
        regions = []

        while lines_to_cover:
            l_to_cover = next(iter(lines_to_cover))
            matching_bites = [(i, b) for i, b in enumerate(bites) if l_to_cover in b]
            if len(matching_bites) != 1:
                raise UnsupportedLayoutError(f'There are multiple regions containing line {l_to_cover}')

            b_id, b = matching_bites[0]
            b = set(b)
            clean = b.issubset(lines_to_cover)
            lines_to_cover -= b
            regions.append((b_id, clean))

        return regions

    def get_bite_reading_order(self, region_centers, bite_regions):
        y_sorted_regions = sorted(bite_regions, key=lambda r: region_centers[r][1])

        # get significantly y-overlapped regions
        # and sort them by x-axis (often a split header line or similar)
        yx_sorted_regions = []

        running_y_overlapped = None
        for r in y_sorted_regions:
            if running_y_overlapped is None:
                running_y_overlapped = [r]
                continue

            if region_centers[r][1] < region_centers[running_y_overlapped[-1]][1] + self.y_margin:
                running_y_overlapped.append(r)
            else:
                yx_sorted_regions.extend(sorted(running_y_overlapped, key=lambda r: region_centers[r][0]))
                running_y_overlapped = [r]

        yx_sorted_regions.extend(sorted(running_y_overlapped, key=lambda r: region_centers[r]))

        bite_reading_order = [r for r in yx_sorted_regions]

        assert len(bite_regions) == len(bite_reading_order)
        return bite_reading_order

    def get_reading_order_xml(self, reading_order, layout):
        reading_order_root = ET.Element("ReadingOrder")
        unordered_root = ET.SubElement(reading_order_root, "UnorderedGroup", attrib={'id': 'root'})
        for bite_id, bite in enumerate(reading_order):
            xml_bite = ET.SubElement(unordered_root, "OrderedGroup", attrib={'id': f'bite_{bite_id+1}'})
            for i, region_id in enumerate(bite):
                ET.SubElement(xml_bite, 'RegionRefIndexed', attrib={'regionRef': layout.regions[region_id].id, 'index': str(i)})

        return reading_order_root

    def ensure_unique_line_ids(self, layout):
        all_line_ids = [line.id for r in layout.regions for line in r.lines]

        if len(all_line_ids) != len(set(all_line_ids)):
            for region in layout.regions:
                for line in region.lines:
                    line.id = f'{region.id}_{line.id}'
