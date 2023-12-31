from enum import Enum
from typing import List, Dict


CZERT_PATH = r"UWB-AIR/Czert-B-base-cased"


class LineLabel(Enum):
    NONE = 0
    TERMINATING = 1
    TITLE = 2


def get_line_clusters(bites: List[List[str]]) -> Dict[str, int]:
    return {line_id: bite_id for bite_id, bite in enumerate(bites) for line_id in bite["lines"]}


def hash_strings(s1: str, s2: str) -> str:
    return s1 + s2 if s1 < s2 else s2 + s1


VALIDATION_FILENAMES_BOOK = [
    "brnensky-drak-02.jpg",
    "ceske-statni-pravo-10.jpg",
    "duverna-chvile-07.jpg",
    "hudba-vecneho-zivota-02.jpg",
    "hvezdne-sny-02.jpg",
    "ivuv-roman-03.jpg",
    "japonsko-a-jeho-lid-02.jpg",
    "kralovna-viktorie-06.jpg",
    "maly-jenik-04.jpg",
    "myslenky-a-nazory-02.jpg",
    "neznamy-host-02.jpg",
    "nikola-suhaj-loupeznik-06.jpg",
    "pocatkove-logiky-02.jpg",
    "pohadka-maje-09.jpg",
    "z-lesni-rise-09.jpg",
]

VALIDATION_FILENAMES_DICTIONARY = [
    "domaci-vseved-07.jpg",
    "domaci-vseved-1699213645-03.jpg",
    "domaci-vseved-1699213645-09.jpg",
    "filosoficky-slovnik-02.jpg",
    "filosoficky-slovnik-1699213638-33.jpg",
    "filosoficky-slovnik-10.jpg",
    "hospodarsky-slovnik-naucny-02.jpg",
    "hospodarsky-slovnik-naucny-06.jpg",
    "hospodarsky-slovnik-naucny-12.jpg",
    "hudebni-slovnik-02.jpg",
    "hudebni-slovnik-04.jpg",
    "hudebni-slovnik-1699213575-14.jpg",
    "prirucni-slovnik-anglicko-cesky-02.jpg",
    "prirucni-slovnik-anglicko-cesky-1699213665-07.jpg",
    "prirucni-slovnik-jazyka-ceskeho-02.jpg",
    "prirucni-slovnik-jazyka-ceskeho-1699213606-12.jpg",
    "prirucny-slovnik-vseobecnych-vedomosti.-(1)-02.jpg",
    "prirucny-slovnik-vseobecnych-vedomosti.-(1)-1699213631-07.jpg",
    "psychologicky-slovnik-04.jpg",
    "psychologicky-slovnik-1699213622-10.jpg",
    "slovnik-francouzsko-cesky.-02.jpg",
    "slovnik-francouzsko-cesky-1699213590-11.jpg",
    "slovnik-naucny-(1)-02.jpg",
    "slovnik-naucny-(1)-1699213671-22.jpg",
    "slovnik-umeni-kucharskeho-02.jpg",
    "slovnik-umeni-kucharskeho-1699213653-06.jpg",
    "slovnik-zdravotni-04.jpg",
    "slovnik-zdravotni-1699213614-03.jpg",
    "strucny-vseobecny-slovnik-vecny-02.jpg",
    "strucny-vseobecny-slovnik-vecny-1699213598-18.jpg",
]


VALIDATION_FILENAMES_PERIODICAL = [
    "lidove-noviny-(1)-02.jpg",
    "lidove-noviny-(1)-06.jpg",
    "lidove-noviny-(6)-1699212980-15.jpg",
    "delnicke-listy-(1)-1699212913-2.jpg",
    "delnicke-listy-(3)-7.jpg",
    "delnicke-listy-(5)6-1699213002-05.jpg",
    "vestnik-obecni-kralovskeho-hlavniho-mesta-prahy-(2)-02.jpg",
    "vestnik-obecni-kralovskeho-hlavniho-mesta-prahy-(12)-1699212475-8.jpg",
    "vestnik-obecni-kralovskeho-hlavniho-mesta-prahy-(13)-1699212957-4.jpg",
    "venkov-(15)-1699212557-02.jpg",
    "venkov-(17)-1699213403-03.jpg",
    "venkov-(20)-1699213145-07.jpg",
    "brnenske-noviny-(1)-1699212832-2.jpg",
    "brnenske-noviny-(8)-1699212991-4.jpg",
    "cesky-zapad-(1)-1699212718-2.jpg",
    "cesky-zapad-(3)-1699213398-5.jpg",
    "druzstevnik-(1)-1699212937-2.jpg",
    "druzstevnik-(5)-1699213267-5.jpg",
    "hlas-od-loucne-(1)-1699212860-2.jpg",
    "hlas-od-loucne-(4)-1699213501-5.jpg",
    "hospodarske-noviny-(2)-1699213123-3.jpg",
    "hospodarske-noviny-(5)-1699213220-7.jpg",
    "listy-zahradnicke-(4)-1699212826-12.jpg",
    "listy-zahradnicke-(5)-1699213514-08.jpg",
    "maj-(1)-1699213132-07.jpg",
    "maj-(4)-1699213084-15.jpg",
    "novy-kadernik-(1)-1699212479-04.jpg",
    "novy-kadernik-(6)-1699212576-7.jpg",
    "obzor-literarni-a-umelecky-(1)-1699213309-11.jpg",
    "obzor-literarni-a-umelecky-(2)-1699212757-16.jpg",
    "ozvena-1699213120-7.jpg",
    "plzenske-listy-(1)-3.jpg",
    "plzenske-listy-(2)-6.jpg",
    "pritomnost-(1)-1699213378-05.jpg",
    "prumyslnik-(1)-1699213546-12.jpg",
    "sekretar-(1)-1699213291-07.jpg",
    "slavie-(1)-1699213059-8.jpg",
    "svetlo-(2)-1699213083-5.jpg",
    "uredni-list-republiky-ceskoslovenske-=-(2)-02.jpg",
    "vestnik-ministerstva-skolstvi-a-kultury-(1)-6.jpg",
    "vlast-(1)-1699213537-3.jpg",
    "zapadocesky-posel-lidu-(1)-4.jpg",
    "zensky-list-(1)-4.jpg",
    "zizka-(1)-7.jpg",
    "zprava-skolni-za-rok-(2)-1699213048-5.jpg",
]


TEST_FILENAMES_BOOK = [
    "brnensky-drak-03.jpg",
    "ceske-statni-pravo-03.jpg",
    "duverna-chvile-09.jpg",
    "hudba-vecneho-zivota-05.jpg",
    "hvezdne-sny-03.jpg",
    "ivuv-roman-05.jpg",
    "japonsko-a-jeho-lid-03.jpg",
    "kralovna-viktorie-07.jpg",
    "maly-jenik-03.jpg",
    "myslenky-a-nazory-04.jpg",
    "neznamy-host-03.jpg",
    "nikola-suhaj-loupeznik-08.jpg",
    "pocatkove-logiky-04.jpg",
    "pohadka-maje-10.jpg",
    "z-lesni-rise-04.jpg",
]


TEST_FILENAMES_DICTIONARY = [
    "domaci-vseved-1699213645-36.jpg",
    "domaci-vseved-1699213645-26.jpg",
    "domaci-vseved-09.jpg",
    "filosoficky-slovnik-1699213638-12.jpg",
    "filosoficky-slovnik-1699213638-16.jpg",
    "filosoficky-slovnik-1699213638-18.jpg",
    "hospodarsky-slovnik-naucny-15.jpg",
    "hospodarsky-slovnik-naucny-1699213569-06.jpg",
    "hospodarsky-slovnik-naucny-1699213569-30.jpg",
    "hudebni-slovnik-1699213575-16.jpg",
    "hudebni-slovnik-03.jpg",
    "hudebni-slovnik-10.jpg",
    "prirucni-slovnik-anglicko-cesky-1699213665-36.jpg",
    "prirucni-slovnik-anglicko-cesky-14.jpg",
    "prirucni-slovnik-jazyka-ceskeho-1699213606-15.jpg",
    "prirucni-slovnik-jazyka-ceskeho-11.jpg",
    "prirucny-slovnik-vseobecnych-vedomosti.-(1)-12.jpg",
    "prirucny-slovnik-vseobecnych-vedomosti.-(1)-1699213631-10.jpg",
    "psychologicky-slovnik-1699213622-09.jpg",
    "psychologicky-slovnik-1699213622-30.jpg",
    "slovnik-francouzsko-cesky-1699213590-16.jpg",
    "slovnik-francouzsko-cesky-1699213590-31.jpg",
    "slovnik-naucny-(1)-1699213671-25.jpg",
    "slovnik-naucny-1699213583-30.jpg",
    "slovnik-umeni-kucharskeho-1699213653-07.jpg",
    "slovnik-umeni-kucharskeho-17.jpg",
    "slovnik-zdravotni-17.jpg",
    "slovnik-zdravotni-1699213614-31.jpg",
    "strucny-vseobecny-slovnik-vecny-1699213598-29.jpg",
    "strucny-vseobecny-slovnik-vecny-06.jpg",
]


TEST_FILENAMES_PERIODICAL = [
    "lidove-noviny-(17)-1699212470-2.jpg",
    "lidove-noviny-(18)-1699212613-3.jpg",
    "lidove-noviny-(20)-1699213534-5.jpg",
    "delnicke-listy-(6)-2.jpg",
    "delnicke-listy-(6)7-1699213363-05.jpg",
    "delnicke-listy-(8)9-1699213505-4.jpg",
    "vestnik-obecni-kralovskeho-hlavniho-mesta-prahy-(1)-02.jpg",
    "vestnik-obecni-kralovskeho-hlavniho-mesta-prahy-(6)-1699213178-12.jpg",
    "vestnik-obecni-kralovskeho-hlavniho-mesta-prahy-(10)-1699212720-18.jpg",
    "venkov-(14)-1699212820-02.jpg",
    "venkov-(11)-1699212483-36.jpg",
    "venkov-(2)-1699213452-07.jpg",
    "brnenske-noviny-(8)-1699212991-2.jpg",
    "brnenske-noviny-1699212880-3.jpg",
    "cesky-zapad-2.jpg",
    "cesky-zapad-4.jpg",
    "druzstevnik-(8)-1699212585-2.jpg",
    "druzstevnik-(7)-1699213039-11.jpg",
    "hlas-od-loucne-(6)-1699212473-2.jpg",
    "hlas-od-loucne-(8)-1699213526-5.jpg",
    "hospodarske-noviny-(6)-1699212964-7.jpg",
    "hospodarske-noviny-(15)-1699213486-7.jpg",
    "listy-zahradnicke-(7)-1699213029-13.jpg",
    "listy-zahradnicke-(6)-1699213224-11.jpg",
    "maj-(5)-1699213104-10.jpg",
    "maj-1699213212-17.jpg",
    "novy-kadernik-1699213269-09.jpg",
    "novy-kadernik-1699213269-02.jpg",
    "obzor-literarni-a-umelecky-(2)-1699212757-14.jpg",
    "obzor-literarni-a-umelecky-1699213507-14.jpg",
    "ozvena-(1)-1699213063-6.jpg",
    "plzenske-listy-3.jpg",
    "plzenske-listy-(3)-2.jpg",
    "pritomnost-(2)-1699212501-10.jpg",
    "prumyslnik-(1)-1699213546-17.jpg",
    "sekretar-(5)-1699212456-12.jpg",
    "slavie-(5)-1699213161-8.jpg",
    "svetlo-(9)-1699213375-6.jpg",
    "uredni-list-republiky-ceskoslovenske-=-(3)-02.jpg",
    "vestnik-ministerstva-skolstvi-a-kultury-5.jpg",
    "vlast-(3)-9.jpg",
    "zapadocesky-posel-lidu-5.jpg",
    "zensky-list-(1)-1699212596-7.jpg",
    "zizka-(1)-1699212599-5.jpg",
    "zprava-skolni-za-rok-(3)-1699213200-7.jpg",
]


VALIDATION_FILENAMES = VALIDATION_FILENAMES_BOOK + VALIDATION_FILENAMES_DICTIONARY + VALIDATION_FILENAMES_PERIODICAL


TEST_FILENAMES = TEST_FILENAMES_BOOK + TEST_FILENAMES_DICTIONARY + TEST_FILENAMES_PERIODICAL


FILENAMES_EXCLUDED_FROM_TRAINING = VALIDATION_FILENAMES + TEST_FILENAMES


if __name__ == "__main__":
    print(f"Total excluded:   {len(FILENAMES_EXCLUDED_FROM_TRAINING)}")
    print(f"Validation total: {len(VALIDATION_FILENAMES)}")
    print(f"Validation books: {len(VALIDATION_FILENAMES_BOOK)}")
    print(f"Validation dicts: {len(VALIDATION_FILENAMES_DICTIONARY)}")
    print(f"Validation perio: {len(VALIDATION_FILENAMES_PERIODICAL)}")
    print(f"Test total:       {len(TEST_FILENAMES)}")
    print(f"Test books:       {len(TEST_FILENAMES_BOOK)}")
    print(f"Test dicts:       {len(TEST_FILENAMES_DICTIONARY)}")
    print(f"Test perio:       {len(TEST_FILENAMES_PERIODICAL)}")

    import os
    for excluded_filename in FILENAMES_EXCLUDED_FROM_TRAINING:
        flag = False
        for root, _, filenames in os.walk(r"/home/martin/textbite/valtest"):
            if excluded_filename in filenames:
                flag = True

        if not flag:
            print(excluded_filename)
    