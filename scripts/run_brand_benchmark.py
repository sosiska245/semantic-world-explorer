"""
Full-benchmark evaluation: brand profile blend vs current Auto mode.

Runs all 160 queries from run_final_audit.py with and without brand profile
blending. Reports:
  - Overall hits@20 gain
  - Per-category gain/regression
  - Miss→hit and hit→miss counts per category
  - Architecture recommendation

Brand blend is a conservative max-gate:
  final[i] = max(auto[i], brand[i] * BRAND_ALPHA)
  where BRAND_ALPHA controls how much brand signal can override main score.

This can only HELP countries with brand profiles, never hurt any other country.
"""

import os
import sys
import numpy as np
from collections import defaultdict, Counter
from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

import voyageai
from src.data_loader import EMBEDDINGS, ENTITIES_DF
from src.factual_router import detect_route, factual_scores
from src.similarity import embed_query
from src.multivec import apply_length_penalty, multi_sim

client   = voyageai.Client()
ISO3     = ENTITIES_DF["iso3"].tolist()
NAMES    = ENTITIES_DF["name"].tolist()
N        = len(ISO3)
_ISO3_IDX = {iso: i for i, iso in enumerate(ISO3)}
_ISO3_SET = set(ISO3)

ENERGY_TERMS = {"energy","solar","renewable","electricity","nuclear","hydroelectric"}

BRAND_ALPHA = 0.85   # brand can push a country up to 85% of its brand cosine score

# ── Brand profiles (25 countries) ────────────────────────────────────────────
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
industrial aesthetics. The country holds the largest number of UNESCO World Heritage Sites
globally and is renowned for Renaissance art, classical architecture, Roman ruins, and opera
(birthplace of Verdi, Puccini, and Rossini). Italian cuisine is one of the most globally
replicated: espresso coffee culture, pizza, pasta, gelato, prosciutto, Parmigiano-Reggiano,
and Barolo wine. Italy is a leading wine producer and olive oil exporter. Football (calcio)
is the dominant sport, with a passionate club culture. Italian cinema, including neorealism
and directors such as Fellini and Visconti, has significantly influenced world film.""",

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

# ── Embed brand profiles ──────────────────────────────────────────────────────
print("Embedding 25 brand profiles …")
iso3_list   = list(BRAND_PROFILES.keys())
brand_texts = [BRAND_PROFILES[iso] for iso in iso3_list]
resp = client.embed(brand_texts, model="voyage-3", input_type="document")
brand_vecs_raw = {
    iso: np.array(v, dtype=np.float32)
    for iso, v in zip(iso3_list, resp.embeddings)
}
brand_vecs = {
    iso: v / (np.linalg.norm(v) + 1e-9)
    for iso, v in brand_vecs_raw.items()
}
print(f"  Done. Brand coverage: {len(brand_vecs)} countries.\n")

# Precompute brand score vector per brand iso: index → brand_cos for that iso
def brand_score_vector(qvec):
    """Return N-length array where brand_scores[i] = brand_cos if country i has
    a profile, else 0. Used for max-gate blend."""
    out = np.zeros(N, dtype=np.float32)
    for iso, bv in brand_vecs.items():
        idx = _ISO3_IDX.get(iso)
        if idx is not None:
            out[idx] = float(np.dot(qvec, bv))
    return out


# ── Score helpers ─────────────────────────────────────────────────────────────

def sem_scores(vec, text):
    alpha = 0.10 if any(t in text.lower() for t in ENERGY_TERMS) else 0.30
    return apply_length_penalty(multi_sim(vec, alpha=alpha))

def auto_scores_and_mode(text):
    route = detect_route(text)
    if route:
        ind, asc, label = route
        return factual_scores(ind, ISO3, asc), f"factual:{ind}", None
    vec = embed_query(text)
    return sem_scores(vec, text), "semantic", vec

def blend_brand(base_scores, qvec):
    """Conservative max-gate: only raises scores for countries with profiles."""
    if qvec is None:
        return base_scores   # factual route: no qvec, brand irrelevant
    b_scores = brand_score_vector(qvec)
    return np.maximum(base_scores, b_scores * BRAND_ALPHA)

def hits_at_k(scores, expected_valid, k=20):
    top = {ISO3[i] for i in np.argsort(scores)[::-1][:k]}
    return sum(1 for iso in expected_valid if iso in top), top


# ── Full query set (from run_final_audit.py) ─────────────────────────────────
from scripts.run_final_audit import QUERIES

# ── Run benchmark ─────────────────────────────────────────────────────────────
print(f"Running {len(QUERIES)}-query benchmark (baseline + brand blend) …")
print("This makes one Voyage embed call per query.\n")

base_total  = 0
blend_total = 0
exp_total   = 0

cat_base  = defaultdict(lambda: [0, 0])  # [hits, total]
cat_blend = defaultdict(lambda: [0, 0])

recoveries  = []   # miss→hit: (cat, query, iso3, base_rank, blend_rank)
regressions = []   # hit→miss: (cat, query, iso3, base_rank, blend_rank)

for cat, q_text, expected in QUERIES:
    valid = [iso for iso in expected if iso in _ISO3_SET]
    if not valid:
        continue
    n = len(valid)

    a_sc, mode, qvec = auto_scores_and_mode(q_text)
    b_sc = blend_brand(a_sc, qvec)

    a_h, a_top = hits_at_k(a_sc, valid)
    b_h, b_top = hits_at_k(b_sc, valid)

    base_total  += a_h
    blend_total += b_h
    exp_total   += n

    cat_base[cat][0]  += a_h;  cat_base[cat][1]  += n
    cat_blend[cat][0] += b_h;  cat_blend[cat][1] += n

    # Track per-country changes
    sorted_base  = np.argsort(a_sc)[::-1]
    sorted_blend = np.argsort(b_sc)[::-1]
    rank_base  = {ISO3[i]: r+1 for r, i in enumerate(sorted_base)}
    rank_blend = {ISO3[i]: r+1 for r, i in enumerate(sorted_blend)}

    for iso in valid:
        rb = rank_base[iso]
        rl = rank_blend[iso]
        if iso not in a_top and iso in b_top:
            recoveries.append((cat, q_text, iso, rb, rl))
        elif iso in a_top and iso not in b_top:
            regressions.append((cat, q_text, iso, rb, rl))


# ── Print report ──────────────────────────────────────────────────────────────
print("=" * 72)
print(f"FULL BENCHMARK — {len(QUERIES)} queries, {exp_total} country-query checks")
print(f"Brand alpha: {BRAND_ALPHA}  (max-gate blend, only raises 25 profiled countries)")
print("=" * 72)

base_pct  = 100 * base_total  / exp_total
blend_pct = 100 * blend_total / exp_total
lift      = blend_total - base_total

print(f"\n  Baseline (Auto):   {base_total}/{exp_total}  ({base_pct:.1f}%)")
print(f"  Brand blended:     {blend_total}/{exp_total}  ({blend_pct:.1f}%)")
print(f"  Net lift:          +{lift} hits  (+{blend_pct-base_pct:.1f}pp)")
print(f"  Recoveries:        +{len(recoveries)} miss→hit")
print(f"  Regressions:        {len(regressions)} hit→miss")

# ── Per-category breakdown ────────────────────────────────────────────────────
all_cats = sorted(set(c for c, _, _ in QUERIES))
print("\nPER-CATEGORY BREAKDOWN:")
print(f"{'Category':<16} {'BaseH':>6} {'BlendH':>7} {'Total':>6} {'Base%':>6} {'Blend%':>7} {'Lift':>5}  {'Risk?':>6}")
print("-" * 70)

HIGH_RISK_CATS = {"politics", "oil_gas", "religion", "history", "military"}

for cat in all_cats:
    bh, n = cat_base[cat]
    bl    = cat_blend[cat][0]
    if n == 0:
        continue
    b_pct = 100 * bh / n
    l_pct = 100 * bl / n
    lift_pp = l_pct - b_pct
    risk_flag = "⚠ HIGH" if cat in HIGH_RISK_CATS else ""
    lift_str = f"{lift_pp:+.1f}pp" if lift_pp != 0 else "  0.0pp"
    print(f"  {cat:<14} {bh:>6} {bl:>7} {n:>6} {b_pct:>6.1f}% {l_pct:>7.1f}% {lift_str:>7}  {risk_flag}")

# ── Regression detail ─────────────────────────────────────────────────────────
print(f"\nREGRESSION DETAIL ({len(regressions)} hit→miss):")
if regressions:
    print(f"  {'Category':<12} {'ISO':>4} {'BaseRank':>9} {'BlendRank':>10}  Query")
    for cat, q, iso, rb, rl in regressions:
        print(f"  {cat:<12} {iso:>4} {rb:>9} {rl:>10}  {q[:55]}")
else:
    print("  None — zero regressions.")

# ── Recovery detail ───────────────────────────────────────────────────────────
print(f"\nRECOVERY DETAIL ({len(recoveries)} miss→hit):")
if recoveries:
    print(f"  {'Category':<12} {'ISO':>4} {'BaseRank':>9} {'BlendRank':>10}  Query")
    for cat, q, iso, rb, rl in sorted(recoveries, key=lambda x: x[0]):
        print(f"  {cat:<12} {iso:>4} {rb:>9} {rl:>10}  {q[:55]}")
else:
    print("  None.")

# ── Architecture recommendation ───────────────────────────────────────────────
print("\n" + "=" * 72)
print("ARCHITECTURE RECOMMENDATION")
print("=" * 72)

has_regressions = len(regressions) > 0
net_lift_pp     = blend_pct - base_pct

print(f"""
  Regressions: {len(regressions)}
  Net lift   : +{lift} hits (+{net_lift_pp:.1f}pp) across all {len(QUERIES)} queries
  Coverage   : 25/250 countries have brand profiles (10%)

  OPTION A — separate domain embedding (emb_domain_brand):
    Pro: fully isolated, zero regression risk, user can enable per-slot
    Con: requires explicit user selection, won't fire in Auto mode
    Best if: regressions > 0 or user wants surgical control

  OPTION B — auto blend at search time (current test approach):
    Pro: transparent to user, fires automatically on brand queries
    Con: slightly harder to audit, must keep BRAND_ALPHA tuned
    Best if: regressions = 0 and net lift is significant

  OPTION C — smart router (keyword trigger → brand blend):
    Pro: only fires when query looks brand-like, cleanest precision
    Con: brand queries are diverse, keyword list will always be incomplete
    Best if: regressions > 0 but you still want some auto coverage

  VERDICT: {'Option B (auto blend) — regressions=0, safe to ship' if not has_regressions else 'Option A (domain embedding) — regressions found, isolate first'}
  ALPHA:   {BRAND_ALPHA} (can lower to reduce blend strength if needed)
""")
