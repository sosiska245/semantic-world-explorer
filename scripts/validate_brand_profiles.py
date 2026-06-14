"""
Validation study: do Brand/Association Profiles produce meaningfully different
signal from the existing Culture section embeddings?

Method:
  1. Hand-craft 25 country brand profiles (~100-200 words, neutral, multi-dimensional)
  2. Embed them via Voyage
  3. For ~30 brand-type queries, compare cosine similarity:
       brand_profile vs emb_culture vs main embedding
  4. Measure:
       (a) Internal similarity: brand ↔ culture (high = redundant, low = complementary)
       (b) Query cosine: does brand >> culture for relevant queries?
       (c) Simulated hits@20: blend brand into existing scores, measure lift
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import voyageai
from src.data_loader import ENTITIES_DF, EMBEDDINGS, SECTION_EMBEDDINGS
from src.similarity import embed_query
from src.multivec import apply_length_penalty, multi_sim

client = voyageai.Client()

# ── 25 brand profiles (neutral, multi-dimensional, source-grounded) ───────────
BRAND_PROFILES = {

"JPN": """Japan is internationally recognized for its contributions to consumer electronics,
automotive engineering, and robotics. Toyota, Sony, Nintendo, and Honda are among the
world's most recognized brands originating here. Japan has a globally influential
entertainment industry spanning anime, manga, video games, and J-pop music. Japanese
cuisine — including sushi, ramen, tempura, and matcha — has spread worldwide and earned
more Michelin-starred restaurants than any other country. Japan is also known for Shinto
and Buddhist heritage, traditional arts such as ikebana and origami, tea ceremony culture,
martial arts including judo and karate, and distinctive architectural and garden aesthetics.
High-speed rail infrastructure, electronics manufacturing precision, and semiconductor
materials production are key economic strengths. Major tourism landmarks include Mount Fuji,
Kyoto temples, and Osaka street food culture.""",

"KOR": """South Korea is internationally recognized for its entertainment industry, including
K-pop music — a global genre with acts such as BTS and BLACKPINK — Korean drama (K-drama),
and webtoon comics. The country is a leading exporter of consumer electronics and
semiconductors through companies such as Samsung and SK Hynix, and home to major automotive
brands including Hyundai and Kia. Korean cuisine, including bibimbap, kimchi, Korean BBQ,
and Korean fried chicken, has gained significant international popularity. South Korea is
known for its high internet penetration, professional esports leagues, and mobile gaming
culture. Traditional cultural heritage includes Confucian philosophy, hanbok clothing,
haenyeo diving traditions, and UNESCO-listed intangible practices. The country achieved
rapid industrial development — often referenced as the Miracle on the Han River — and has
strong design, film, and beauty (K-beauty) export industries.""",

"ITA": """Italy is internationally associated with fashion design, luxury goods, and haute couture,
home to fashion houses including Gucci, Prada, Versace, Armani, and Valentino. Italian
design extends to furniture, automotive styling (Ferrari, Lamborghini, Alfa Romeo), and
industrial aesthetics (La Rinascente, Fiat). The country holds the largest number of
UNESCO World Heritage Sites globally and is renowned for Renaissance art, classical
architecture, Roman ruins, and opera (birthplace of Verdi, Puccini, and Rossini). Italian
cuisine is one of the most globally replicated: espresso coffee culture, pizza, pasta,
gelato, prosciutto, Parmigiano-Reggiano, and Barolo wine. Italy is a leading wine producer
and olive oil exporter. Football (calcio) is the dominant sport, with a passionate club
culture. Italian cinema, including neorealism and directors such as Fellini and Visconti,
has significantly influenced world film.""",

"FRA": """France is internationally recognized as a global centre of fashion, luxury goods, and
haute couture, home to Louis Vuitton, Chanel, Hermès, Dior, and Yves Saint Laurent. French
cuisine holds UNESCO Intangible Heritage status and is associated with wine (Bordeaux,
Burgundy, Champagne appellations), cheese diversity (over 400 varieties), bistro culture,
and Michelin-starred gastronomy. France is the world's most visited tourist destination,
with Paris landmarks including the Eiffel Tower, the Louvre museum, and Versailles. French
cinema (Cannes Film Festival, nouvelle vague movement) and literature (Camus, Sartre, Hugo,
Proust) have been globally influential. France has strong traditions in classical ballet,
opera, and fine art. Cycling culture includes hosting the Tour de France annually. France
is a nuclear energy leader, aerospace manufacturer (Airbus), and permanent UN Security
Council member with one of the world's largest nuclear fleets.""",

"DEU": """Germany is internationally recognized for engineering excellence, automotive manufacturing
(Volkswagen, BMW, Mercedes-Benz, Porsche), and precision machinery exports. It is Europe's
largest economy and a global leader in industrial automation, chemical production (BASF),
and pharmaceutical manufacturing (Bayer). Beer culture is an important heritage element,
with the Reinheitsgebot purity law and the Oktoberfest festival internationally recognized.
Germany has one of the world's richest classical music traditions — birthplace of Bach,
Beethoven, Brahms, and Wagner — and hosts major opera houses and symphony orchestras.
The country has a strong contemporary art scene, design tradition (Bauhaus movement),
and literary heritage (Goethe, Schiller, Kafka). Football (Bundesliga) is the dominant
sport. Germany is a leader in renewable energy transition (Energiewende), wind power
infrastructure, and environmental policy. Berlin has emerged as a major techno music and
startup ecosystem hub.""",

"CHE": """Switzerland is internationally recognized for private banking, wealth management, and
financial services centred in Zurich and Geneva. Swiss watchmaking — including brands such
as Rolex, Omega, Patek Philippe, and Swatch — is a global heritage industry. Swiss
chocolate production (Lindt, Nestlé, Toblerone) and cheese traditions (Gruyère, Emmental,
raclette, fondue) are strongly associated with the country internationally. Switzerland hosts
major international institutions including the Red Cross (founded in Geneva), WHO, WTO, and
UN European headquarters. Alpine tourism, skiing resorts (Davos, Zermatt, St. Moritz), and
mountaineering heritage (Matterhorn, Alps) are significant. Swiss pharmaceutical and
biotechnology companies (Novartis, Roche) are global leaders. The country is known for
political neutrality, direct democracy, and multi-lingual identity (German, French, Italian,
Romansh). CERN, the particle physics laboratory, is located on the Swiss-French border.""",

"BRA": """Brazil is internationally associated with football — the country has won the FIFA World
Cup five times and produced players including Pelé and Ronaldo. Brazilian Carnival, held
annually in Rio de Janeiro and Salvador, is one of the world's largest festivals, featuring
samba dance and music traditions. Brazil contains the Amazon rainforest, the world's largest
tropical forest and a major global biodiversity reserve. The country is the world's largest
producer and exporter of coffee and a leading producer of sugar cane, soybeans, and orange
juice. Capoeira (a martial art-dance fusion) and bossa nova music are internationally
recognized Brazilian cultural exports. Brazilian cinema, literature (Jorge Amado, Clarice
Lispector), and architecture (Oscar Niemeyer, Brasília) have international recognition.
Brazil's aviation industry (Embraer) and agricultural technology sector are major economic
strengths. Brazilian Jiu-Jitsu has spread globally as a martial art.""",

"USA": """The United States is internationally recognized as the centre of the global entertainment
industry, with Hollywood film production, the music recording industry (jazz, blues, rock,
hip-hop, country originating here), and major television and streaming platforms. Silicon
Valley is the world's leading technology and venture capital hub, home to Apple, Google,
Amazon, Meta, and Microsoft. American universities including MIT, Harvard, Stanford, and
Caltech rank among the world's most influential research institutions. NASA space exploration
milestones, including the Apollo lunar missions, are globally recognized. US cultural exports
include fast food brands (McDonald's, Coca-Cola), denim fashion, and professional sports
leagues (NBA, NFL, MLB). The country has the world's largest national economy by GDP. New
York City is a global financial centre (NYSE, NASDAQ). American literature, from Mark Twain
to Toni Morrison, and visual art (Abstract Expressionism) have shaped global culture.""",

"IND": """India is internationally recognized for its film industry centred in Mumbai — Bollywood
produces more films annually than Hollywood and has global audiences. Indian classical music
(Hindustani and Carnatic traditions), dance forms (Bharatanatyam, Kathak), and yoga have
spread internationally and gained UNESCO recognition. Indian cuisine is globally recognized:
spice diversity, vegetarian culinary traditions, tandoor cooking, biryani, dal, and chai tea
culture have strong international presence. Cricket is the dominant sport, with the Indian
Premier League (IPL) one of the world's most watched sporting events. India has a major IT
and software services export sector centred in Bangalore, Hyderabad, and Chennai (companies
including Infosys, Wipro, TCS). Traditional crafts include handloom weaving, pottery,
jewellery, and Mughal-era architecture (Taj Mahal). Space programme achievements (ISRO
missions to Mars and the Moon) have gained international recognition.""",

"JAM": """Jamaica is internationally recognized as the birthplace and global home of reggae music
— a genre originating in Kingston in the late 1960s, associated with artists including
Bob Marley, Peter Tosh, Jimmy Cliff, and Burning Spear. Reggae's UNESCO Intangible Cultural
Heritage status reflects its global cultural influence. Jamaican music has also contributed
ska, rocksteady, dancehall, and dub genres internationally. Jamaica holds notable records
in track and field athletics, particularly sprinting — Usain Bolt, Shelly-Ann Fraser-Pryce,
and a deep tradition of Olympic sprint champions. Jamaican rum is a significant export.
Blue Mountain coffee is recognized internationally as a premium specialty variety. Jerk
cooking — a distinctive spice-smoke preparation technique — is a recognized element of
Jamaican food culture. Rastafari spiritual movement originated in Jamaica and has spread
globally as a cultural and religious influence.""",

"BEL": """Belgium is internationally recognized as the world's leading centre of specialty beer
production, with over 1,500 distinct beer varieties including Trappist ales (Westvleteren,
Chimay), lambic fermentation, and sour ales; Belgian beer holds UNESCO Intangible Heritage
status. Belgian chocolate (Godiva, Neuhaus, Côte d'Or) is globally recognized as a premium
confectionery tradition. Belgium is the historical home of cycling culture — producing Tour
de France champions and hosting prestigious classics including the Tour of Flanders and
Liège–Bastogne–Liège. Belgium hosts the headquarters of the European Union and NATO in
Brussels. The country has a strong comic art tradition (Tintin, The Smurfs) and international
representation in fine art (Flemish masters: Rubens, Van Dyck; Surrealism: Magritte,
Delvaux). Belgian waffles and frites (french fries) are internationally associated food
exports. Ghent and Bruges are recognized for medieval architecture and cultural heritage.""",

"NLD": """The Netherlands is internationally recognized for cycling infrastructure — with more
bicycles than people and a dense national cycling network, it is the global reference model
for cycling urban planning. Dutch Golden Age painting (Rembrandt, Vermeer, Frans Hals) is
among the world's most studied art traditions; the Rijksmuseum in Amsterdam holds an
internationally significant collection. The country is a major global logistics hub: the
Port of Rotterdam is Europe's largest and one of the busiest globally. Dutch agriculture
is remarkably productive per hectare — second largest food exporter by value globally.
Flower cultivation (tulips, cut flowers) and horticulture are internationally associated
exports. The Netherlands was historically a major maritime trading empire (Dutch East India
Company). Dutch design (Droog, TU Delft), architecture (Rem Koolhaas, MVRDV), and
electronic music (Rotterdam, Amsterdam club scenes) have international recognition. Gouda
and Edam cheeses are internationally recognized Dutch food heritage.""",

"AUT": """Austria is internationally recognized as one of the world's great centres of classical
music — birthplace of Mozart, Schubert, Brahms, Haydn, and Johann Strauss, and home to
the Vienna Philharmonic Orchestra, Vienna State Opera, and Salzburg Music Festival.
Vienna's coffeehouse culture (Viennese café tradition) holds UNESCO Intangible Heritage
status. Skiing and alpine tourism are central economic and cultural activities — major ski
resorts (St. Anton, Kitzbühel, Lech) and alpine heritage including lederhosen and dirndl
traditional dress. Austrian cuisine includes Wiener Schnitzel, Sachertorte, Apfelstrudel,
and Kaiserschmarrn, with strong coffeehouse pastry traditions. Psychoanalysis was
developed in Vienna by Sigmund Freud. Austria is the birthplace of Arnold Schwarzenegger
and has contributed significantly to philosophy (Wittgenstein, Popper), art (Klimt,
Schiele), and architecture. The country is a major centre of humanitarian diplomacy,
hosting UN agencies and international negotiations in Vienna.""",

"CZE": """Czechia (Czech Republic) is internationally recognized as one of the world's great beer
cultures — Prague and Bohemia gave origin to the Pilsner style, and Czech per-capita beer
consumption consistently ranks among the world's highest. Czech Bohemian crystal glassware
is a recognized decorative arts tradition. The country has a rich literary heritage
(Franz Kafka, Milan Kundera, Jaroslav Hašek) and was the centre of the Czechoslovak New
Wave cinema movement internationally recognized in the 1960s. Prague's historic city centre
and Gothic, Baroque, and Art Nouveau architecture draw significant international tourism.
Czech classical music contributions include Antonín Dvořák and Bedřich Smetana. The country
is a significant automotive manufacturing centre (Škoda) and has a developed engineering
sector. Traditional crafts include Bohemian puppetry (UNESCO listed), marionette theatre,
and folk embroidery. Czech spa culture, particularly in Karlovy Vary, has historical
international recognition.""",

"ETH": """Ethiopia is internationally recognized as the origin country of coffee — the plant
Coffea arabica is native to the Ethiopian highlands, and traditional Ethiopian coffee
ceremony (a multi-stage social ritual) holds UNESCO Intangible Heritage recognition.
Ethiopia is among the world's top coffee exporters, with Yirgacheffe, Sidama, and Harrar
varieties prized in specialty coffee markets. Ethiopian long-distance running has produced
an extraordinary concentration of Olympic and World Championship gold medalists including
Abebe Bikila, Haile Gebrselassie, Kenenisa Bekele, and Tirunesh Dibaba. Injera flatbread
and communal stew (wat) dining traditions are internationally recognized elements of
Ethiopian cuisine. The country contains some of Africa's most significant ancient
archaeological and religious sites: Lalibela rock-hewn churches (UNESCO), Aksum obelisks,
the Rift Valley palaeontological sites, and early Christian and Islamic heritage.
Ethiopia has the largest number of UNESCO World Heritage Sites on the African continent.""",

"COL": """Colombia is internationally recognized as the world's premier source of washed Arabica
specialty coffee — Colombian coffee has UNESCO-recognized "coffee cultural landscape" status.
Colombian emerald mining accounts for a significant share of global supply. García Márquez,
who pioneered magical realism literature and won the Nobel Prize in Literature in 1982, is
Colombia's most internationally recognized cultural figure. Colombian cycling has produced
multiple Tour de France climber champions (Nairo Quintana, Egan Bernal, Miguel Ángel López).
Cumbia and vallenato music are internationally recognized Colombian cultural exports.
Colombian salsa dance tradition, centred in Cali, has international competition recognition.
Medellín's urban transformation from industrial city to innovation hub has been internationally
studied in urban development contexts. Colombian cut flower exports (roses, carnations) are
among the world's largest. Cartagena's colonial walled city is a UNESCO World Heritage Site
and significant heritage tourism destination.""",

"ARG": """Argentina is internationally associated with tango — a musical and dance form originating
in Buenos Aires in the late 19th century, now UNESCO Intangible Heritage. Argentine football
has produced globally recognized players including Diego Maradona and Lionel Messi, and
the national team won the 2022 FIFA World Cup. Argentine beef and cattle ranching (gaucho
tradition, asado barbecue culture) are internationally recognized food heritage elements.
Mendoza wine region produces Malbec varieties internationally recognized in fine wine
markets. Argentina has a significant literary tradition (Jorge Luis Borges, Julio Cortázar)
associated with magical realism and the Latin American literary boom. Patagonia and the
Andes mountain range attract significant adventure tourism. Argentina has one of Latin
America's largest university-educated populations and a strong tradition in science and
medicine — with multiple Nobel Prize winners in chemistry, medicine, and peace.""",

"SGP": """Singapore is internationally recognized as a major global aviation hub — Changi Airport
consistently ranks as the world's best airport and is a primary transit point for
Southeast Asian air travel. The country is a leading global financial services centre and
wealth management hub, with one of the world's highest concentrations of private banking
activity. Singapore's street food culture — hawker centres, hainanese chicken rice, laksa,
char kway teow, and chilli crab — has UNESCO Intangible Heritage recognition. Singapore is
a leading port and maritime trade hub, one of the world's busiest container ports.
The country has built a strong reputation for urban garden planning (Gardens by the Bay,
green building integration) and liveable city design. Singapore hosts a regional technology
and fintech hub, with significant Southeast Asian startup investment. Education quality and
PISA rankings consistently place Singapore near the global top. Multinational regional
headquarters concentration in Singapore is among the highest globally.""",

"ISR": """Israel is internationally recognized in technology and innovation sectors — with the
highest concentration of startup companies per capita globally, it is often described
as the 'Startup Nation'. Cybersecurity, agricultural technology (drip irrigation innovation),
and medical device development are areas of particular international recognition. Israeli
research universities (Weizmann Institute, Hebrew University, Technion) have produced
multiple Nobel Prize laureates in chemistry and economics. The Dead Sea, Mediterranean
coastline, Negev desert, and Jerusalem's religious heritage sites (holy to Judaism,
Christianity, and Islam) draw significant international tourism. Israeli cuisine — hummus,
falafel, shakshuka, tahini, and Levantine mezze culture — has spread internationally.
Tel Aviv has an internationally recognized contemporary art, architecture, and nightlife
scene. Traditional arts and crafts, contemporary dance (Batsheva Dance Company), and
classical music (Israel Philharmonic Orchestra) have international standing.""",

"FIN": """Finland is internationally recognized for exceptional education outcomes — the Finnish
school system has served as an international reference model for educational quality and
equity. Finland has one of the world's most active heavy metal music scenes relative to
population size, producing internationally known bands including Nightwish, Children of
Bodom, HIM, and Apocalyptica. Nokia's historical contribution to mobile telecommunications
and Rovio's Angry Birds game franchise are internationally recognized Finnish technology
exports. Finnish design tradition — Iittala glassware, Marimekko textiles, Aalto furniture,
Artek — is a globally recognized modernist design heritage. Sauna culture is intrinsic
to Finnish identity and holds UNESCO Intangible Heritage status. Finland hosts the annual
midnight sun phenomenon in the north and aurora borealis tourism. Finnish rally driving
heritage (Tommi Mäkinen, Marcus Grönholm) and cross-country skiing traditions are
internationally recognized. Sibelius is Finland's most celebrated classical composer.""",

"NZL": """New Zealand is internationally recognized for rugby union — the All Blacks national team
has historically been rated the world's top-ranked team and is associated with the haka
ceremonial performance before matches. The country's dramatic landscapes have global
recognition following their use as filming locations for The Lord of the Rings and The
Hobbit trilogy. Māori culture, including haka, kapa haka performing arts, carving traditions,
and te reo Māori language revitalization, is recognized internationally as an indigenous
cultural heritage. New Zealand wines — particularly Marlborough Sauvignon Blanc — are
internationally recognized in fine wine markets. Adventure tourism (bungee jumping, skydiving,
mountain trekking, glacier hiking) is a significant element of New Zealand's international
tourism identity. New Zealand is recognized for progressive political achievements including
being the first country to grant women's suffrage in 1893. Dairy exports (Fonterra) and
sheep farming are major agricultural sectors. New Zealand's conservation approach to
endemic wildlife (kiwi bird, tuatara) has international recognition.""",

"TWN": """Taiwan is internationally recognized as the world's leading semiconductor and
microchip manufacturing hub — TSMC (Taiwan Semiconductor Manufacturing Company) produces
the majority of the world's most advanced integrated circuits and is considered a critical
global supply chain chokepoint. Taiwan is home to major electronics brands including Asus,
Acer, HTC, and Foxconn (manufacturing partner for Apple and other global brands). Night
market food culture is internationally associated with Taiwan: bubble tea (boba) was
invented in Taichung, and Taiwanese street food including beef noodle soup, scallion
pancakes, and stinky tofu are widely recognized in global food contexts. Traditional Chinese
arts and heritage, including calligraphy, ink painting, and classical music, are
preserved in institutions such as the National Palace Museum. Taiwan has strong indigenous
Austronesian cultural heritage. High-speed rail infrastructure and urban transit systems
are internationally studied as infrastructure examples.""",

"NGA": """Nigeria is internationally recognized as the centre of African film production —
Nollywood is the world's second largest film industry by volume, with significant audiences
across Africa and in diaspora communities globally. Afrobeats music — a fusion of Nigerian
and West African rhythms with contemporary pop influences — has achieved major international
commercial success, with artists including Burna Boy, Wizkid, and Davido reaching global
audiences. Nigerian cuisine including jollof rice, egusi soup, suya, and puff-puff has
international presence in diaspora communities. Nigeria is Africa's largest economy and
a major oil and gas producer (Niger Delta reserves). The country has a rich tradition of
textile arts (Aso-oke woven fabric, Ankara prints), bronze casting heritage from the
Benin Kingdom (Benin Bronzes held in international museum collections), and Yoruba,
Igbo, and Hausa cultural traditions with internationally studied literary outputs
(Chinua Achebe's Things Fall Apart is one of the most widely read African literary works).""",

"MEX": """Mexico is internationally associated with a cuisine recognized by UNESCO as Intangible
Cultural Heritage — including tortillas, tamales, mole sauces, chillies, chocolate, and
guacamole. Tequila (from Jalisco's blue agave) and mezcal spirits are globally recognized
exports. Mariachi music is a UNESCO-listed Mexican musical tradition. Day of the Dead
(Día de los Muertos) celebrations are internationally recognized as a distinct cultural
practice blending indigenous and Catholic traditions. Mexico's ancient civilizations —
Aztec (Tenochtitlán), Maya (Chichén Itzá, Palenque), and other pre-Columbian cultures —
represent significant archaeological heritage. Mexican contemporary art, including muralism
(Diego Rivera, José Clemente Orozco) and Frida Kahlo, has international cultural standing.
Mexico is North America's third largest economy, a major automotive manufacturing hub
(Volkswagen, BMW, General Motors assembly plants), and a significant silver and avocado
exporter. Lucha libre wrestling is a recognized Mexican cultural spectacle internationally.""",

"THA": """Thailand is internationally recognized as one of Asia's premier tourism destinations,
with approximately 40 million international visitors annually pre-pandemic. Thai street food
culture — pad thai, green curry, som tum (papaya salad), tom yum soup, and mango sticky
rice — has spread internationally and Thai restaurants are among the most globally
common ethnic food establishments. Muay Thai (Thai boxing) is an internationally
practiced martial art originating in Thailand, with a significant global competition circuit.
Thailand is a major Buddhist country with significant temple heritage (Wat Phra Kaew, Wat
Pho, Chiang Mai temples) that draws religious and cultural tourism. Thai silk and traditional
handicraft (Celadon ceramics, lacquerware, woodcarving) have heritage export recognition.
The country is a major agricultural exporter of rice, rubber, and tapioca. Thailand's
health and medical tourism sector is internationally recognized for affordable high-quality
procedures. Traditional Thai massage is a UNESCO Intangible Heritage practice.""",

}

# ── Brand query set: queries that should benefit from brand profiles ───────────
# Each entry: (query_text, [expected_iso3_list], category)
BRAND_QUERIES = [
    # Music
    ("anime manga pop culture",                     ["JPN"],            "entertainment"),
    ("K-pop music Korean wave Hallyu",              ["KOR"],            "music"),
    ("reggae music Bob Marley Jamaica",             ["JAM"],            "music"),
    ("heavy metal music Scandinavia Finland",       ["FIN"],            "music"),
    ("tango dance music Buenos Aires",              ["ARG"],            "music"),
    ("classical music opera symphony Vienna",       ["AUT","DEU"],      "music"),
    ("jazz blues soul music American roots",        ["USA"],            "music"),
    ("Afrobeats music Nigeria West Africa",         ["NGA"],            "music"),
    # Food & Drink
    ("espresso coffee cafe culture",                ["ITA","AUT"],      "food"),
    ("specialty coffee origin Ethiopia",            ["ETH","COL"],      "food"),
    ("wine viticulture Bordeaux Merlot",            ["FRA","ITA","ARG"],"food"),
    ("beer craft brewing Trappist Pilsner",         ["BEL","CZE","DEU"],"food"),
    ("chocolate luxury confectionery Swiss",        ["CHE","BEL"],      "food"),
    ("street food hawker night market Asia",        ["THA","SGP","TWN"],"food"),
    # Sports
    ("cycling Tour de France peloton",              ["FRA","BEL","NLD","COL"],"sports"),
    ("cricket bat wicket national sport",           ["IND","AUS"],      "sports"),
    ("rugby union All Blacks haka",                 ["NZL"],            "sports"),
    ("football soccer samba style",                 ["BRA","ARG"],      "sports"),
    # Film & Entertainment
    ("Bollywood Hindi film industry",               ["IND"],            "film"),
    ("Hollywood film studio movie production",      ["USA"],            "film"),
    ("Nollywood African film cinema",               ["NGA"],            "film"),
    ("video games esports gaming culture",          ["JPN","KOR","USA"],"entertainment"),
    # Technology & Innovation
    ("startup nation venture capital unicorn",      ["USA","ISR","SGP"],"tech"),
    ("semiconductor chip foundry TSMC",             ["TWN","KOR"],      "tech"),
    ("automotive engineering precision luxury car", ["DEU","ITA","CHE"],"tech"),
    # Fashion & Design
    ("fashion luxury brands haute couture",         ["FRA","ITA"],      "fashion"),
    ("cycling infrastructure bicycle urban",        ["NLD","BEL"],      "design"),
    ("Bauhaus modernist design architecture",       ["DEU"],            "design"),
    # Cultural Heritage
    ("carnival festival samba Rio",                 ["BRA"],            "culture"),
    ("martial arts Muay Thai kickboxing",           ["THA","JPN"],      "culture"),
]

# ── Embed brand profiles ──────────────────────────────────────────────────────
print("Embedding brand profiles …")
iso3_list   = list(BRAND_PROFILES.keys())
brand_texts = [BRAND_PROFILES[iso] for iso in iso3_list]

resp = client.embed(brand_texts, model="voyage-3", input_type="document")
brand_vecs_raw = {iso: np.array(v, dtype=np.float32) for iso, v in zip(iso3_list, resp.embeddings)}

# Normalise
brand_vecs = {iso: v / (np.linalg.norm(v) + 1e-9) for iso, v in brand_vecs_raw.items()}
print(f"  {len(brand_vecs)} brand profiles embedded.")

# ── Fetch entity indices for each iso3 ────────────────────────────────────────
iso3_to_idx = {row["iso3"]: idx for idx, row in ENTITIES_DF.iterrows()}

# ── Pull existing culture section embeddings ──────────────────────────────────
culture_vecs_raw = SECTION_EMBEDDINGS.get("culture", [])

def get_culture_vec(iso3):
    idx = iso3_to_idx.get(iso3)
    if idx is None or idx >= len(culture_vecs_raw) or culture_vecs_raw[idx] is None:
        return None
    v = culture_vecs_raw[idx]
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None

def get_main_vec(iso3):
    idx = iso3_to_idx.get(iso3)
    if idx is None:
        return None
    v = EMBEDDINGS[idx]
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None

# ── SECTION 1: Internal similarity — brand vs culture ────────────────────────
print("\n" + "=" * 70)
print("SECTION 1 — Internal similarity: brand_profile ↔ culture_section")
print("(high cosine = redundant; low = complementary new signal)")
print("=" * 70)
print(f"{'ISO3':<6} {'Country':<22} {'Brand↔Culture':>14} {'Brand↔Main':>11}")
print("-" * 57)

brand_culture_sims = []
brand_main_sims = []

for iso in iso3_list:
    bv = brand_vecs[iso]
    cv = get_culture_vec(iso)
    mv = get_main_vec(iso)
    bc = float(np.dot(bv, cv)) if cv is not None else None
    bm = float(np.dot(bv, mv)) if mv is not None else None
    name = ENTITIES_DF[ENTITIES_DF["iso3"] == iso]["name"].values
    name_str = name[0] if len(name) else iso
    bc_str = f"{bc:.3f}" if bc is not None else "n/a"
    bm_str = f"{bm:.3f}" if bm is not None else "n/a"
    print(f"{iso:<6} {name_str:<22} {bc_str:>14} {bm_str:>11}")
    if bc is not None:
        brand_culture_sims.append(bc)
    if bm is not None:
        brand_main_sims.append(bm)

print(f"\n  Average brand↔culture cosine : {np.mean(brand_culture_sims):.3f}  (< 0.85 = meaningfully different)")
print(f"  Average brand↔main cosine    : {np.mean(brand_main_sims):.3f}")

# ── SECTION 2: Per-query cosine comparison ────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 2 — Per-query cosine: brand vs culture vs main (for expected countries)")
print("=" * 70)
print(f"{'Category':<14} {'Query':<42} {'ISO':>5} {'Brand':>7} {'Culture':>8} {'Main':>7} {'Winner':>8}")
print("-" * 93)

wins = {"brand": 0, "culture": 0, "main": 0, "tie": 0}
results = []

for query, expected_iso3s, category in BRAND_QUERIES:
    qvec_resp = client.embed([query], model="voyage-3", input_type="query")
    qvec = np.array(qvec_resp.embeddings[0], dtype=np.float32)
    qvec /= np.linalg.norm(qvec) + 1e-9

    for iso in expected_iso3s:
        if iso not in brand_vecs:
            continue
        bv = brand_vecs[iso]
        cv = get_culture_vec(iso)
        mv = get_main_vec(iso)

        b_sim = float(np.dot(qvec, bv))
        c_sim = float(np.dot(qvec, cv)) if cv is not None else -1.0
        m_sim = float(np.dot(qvec, mv)) if mv is not None else -1.0

        best = max(("brand", b_sim), ("culture", c_sim), ("main", m_sim), key=lambda x: x[1])
        wins[best[0]] += 1

        q_short = query[:42]
        print(f"{category:<14} {q_short:<42} {iso:>5} {b_sim:>7.3f} {c_sim:>8.3f} {m_sim:>7.3f} {best[0]:>8}")
        results.append({"query": query, "iso": iso, "category": category,
                         "brand": b_sim, "culture": c_sim, "main": m_sim, "winner": best[0]})

print(f"\n  Brand wins: {wins['brand']}  |  Culture wins: {wins['culture']}  |  Main wins: {wins['main']}")

# ── SECTION 3: Simulated hits@20 with blended brand scores ───────────────────
print("\n" + "=" * 70)
print("SECTION 3 — Simulated hits@20 (all 250 countries, brand blended for 25)")
print("=" * 70)
print("Method: sim_final[i] = main[i] if no brand profile else max(main[i], brand[i]*0.7)")
print("(conservative blend: brand acts as a boost signal, not a replacement)")

N = len(ENTITIES_DF)
all_iso3 = ENTITIES_DF["iso3"].tolist()

hits_main  = 0
hits_brand = 0
total_checks = 0

detail_rows = []

for query, expected_iso3s, category in BRAND_QUERIES:
    qvec_resp = client.embed([query], model="voyage-3", input_type="query")
    qvec = np.array(qvec_resp.embeddings[0], dtype=np.float32)
    qvec /= np.linalg.norm(qvec) + 1e-9

    # Main embedding scores (all 250)
    main_scores = EMBEDDINGS @ qvec  # raw dot product (embeddings are normalised)

    # Brand-blended scores
    blended = main_scores.copy()
    for iso, bv in brand_vecs.items():
        idx = iso3_to_idx.get(iso)
        if idx is None:
            continue
        b_sim = float(np.dot(qvec, bv))
        blended[idx] = max(float(blended[idx]), b_sim * 0.85)

    # Apply length penalty (same as production)
    main_lp     = apply_length_penalty(main_scores)
    blended_lp  = apply_length_penalty(blended)

    main_order   = np.argsort(main_lp)[::-1]
    blended_order = np.argsort(blended_lp)[::-1]

    for iso in expected_iso3s:
        idx = iso3_to_idx.get(iso)
        if idx is None:
            continue
        total_checks += 1

        rank_main   = int(np.where(main_order == idx)[0][0]) + 1
        rank_blend  = int(np.where(blended_order == idx)[0][0]) + 1

        hit_main  = rank_main  <= 20
        hit_blend = rank_blend <= 20
        hits_main  += int(hit_main)
        hits_brand += int(hit_blend)

        detail_rows.append({
            "cat": category, "query": query[:38], "iso": iso,
            "rank_main": rank_main, "rank_blend": rank_blend,
            "hit_main": hit_main, "hit_blend": hit_blend,
        })

print(f"\n{'Category':<14} {'Query':<39} {'ISO':>5} {'R_main':>7} {'R_blend':>8} {'MainHit':>8} {'BrandHit':>9}")
print("-" * 95)
for r in detail_rows:
    m_flag = "✓" if r["hit_main"]  else "✗"
    b_flag = "✓" if r["hit_blend"] else "✗"
    print(f"{r['cat']:<14} {r['query']:<39} {r['iso']:>5} {r['rank_main']:>7} {r['rank_blend']:>8} {m_flag:>8} {b_flag:>9}")

print(f"\n  Total country-query checks : {total_checks}")
print(f"  hits@20  main only         : {hits_main}  ({hits_main/total_checks*100:.1f}%)")
print(f"  hits@20  brand blended     : {hits_brand}  ({hits_brand/total_checks*100:.1f}%)")
print(f"  Lift from brand profiles   : +{hits_brand - hits_main} hits  ({(hits_brand-hits_main)/total_checks*100:+.1f}pp)")

# ── SECTION 4: Brand-specific lift analysis ───────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 4 — Lift breakdown by category")
print("=" * 70)

by_cat = {}
for r in detail_rows:
    c = r["cat"]
    if c not in by_cat:
        by_cat[c] = {"main": 0, "brand": 0, "total": 0}
    by_cat[c]["total"] += 1
    by_cat[c]["main"]  += int(r["hit_main"])
    by_cat[c]["brand"] += int(r["hit_blend"])

print(f"{'Category':<14} {'Total':>6} {'MainHits':>9} {'BrandHits':>10} {'Lift':>6}")
print("-" * 50)
for cat, d in sorted(by_cat.items()):
    lift = d["brand"] - d["main"]
    print(f"{cat:<14} {d['total']:>6} {d['main']:>9} {d['brand']:>10} {lift:>+6}")

# ── SECTION 5: Cases where brand raises a country from miss to hit ─────────────
print("\n" + "=" * 70)
print("SECTION 5 — Miss→Hit conversions (brand recovered a failed query)")
print("=" * 70)
converted = [r for r in detail_rows if not r["hit_main"] and r["hit_blend"]]
degraded  = [r for r in detail_rows if r["hit_main"]  and not r["hit_blend"]]
print(f"  Recoveries (miss→hit) : {len(converted)}")
for r in converted:
    print(f"    [{r['cat']}] {r['query'][:38]} → {r['iso']}  rank {r['rank_main']} → {r['rank_blend']}")
if degraded:
    print(f"\n  Regressions (hit→miss) : {len(degraded)}")
    for r in degraded:
        print(f"    [{r['cat']}] {r['query'][:38]} → {r['iso']}  rank {r['rank_main']} → {r['rank_blend']}")
else:
    print(f"\n  Regressions (hit→miss) : 0 — no degradation")
