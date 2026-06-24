"""
Mock disaster knowledge documents for the RAG knowledge base.
Simulates NDMA flood reports, FFD river stage documents, and historical flood records.
"""

MOCK_DOCUMENTS = [
    {
        "source": "NDMA Flood Report 2010",
        "title": "National Disaster Management Authority — Super Floods 2010: District Situation Reports",
        "content": (
            "The 2010 Pakistan Super Floods were triggered by unprecedented monsoon rainfall "
            "across the Khyber Pakhtunkhwa, Punjab, Sindh, and Balochistan provinces. "
            "Charsadda district in KPK was among the worst-affected areas. The district "
            "experienced severe inundation due to Kabul River overflow, which breached its "
            "embankments on 29 July 2010 following continuous heavy rainfall in the upper "
            "catchment. The overflow led to the displacement of residents from over 200 "
            "villages, with approximately 500,000 people evacuated to relief camps. "
            "Agricultural lands covering 85,000 hectares were submerged, and critical "
            "infrastructure including the Nowshera-Charsadda road was washed away. "
            "The Kabul River at Nowshera gauge recorded a peak discharge of 850,000 cusecs, "
            "the highest since records began. Relief operations were coordinated through "
            "NDMA provincial offices and the Pakistan Army. Rescue teams evacuated "
            "communities from Tangi, Shabqadar, Prang, and Umarzai union councils. "
            "The death toll in Charsadda reached 237 with over 1,200 injuries reported. "
            "Total economic losses for the district were estimated at PKR 42 billion."
        ),
    },
    {
        "source": "NDMA Flood Report 2010",
        "title": "National Disaster Management Authority — Super Floods 2010: Sindh Province Assessment",
        "content": (
            "Sindh province suffered catastrophic flooding during the 2010 Super Floods as "
            "floodwaters moved southward through the Indus River system. Sukkur, Larkana, "
            "Jacobabad, and Shikarpur districts recorded flood extents exceeding their 2005 "
            "benchmarks by 40%. The Indus at Sukkur Barrage recorded inflows of 1.1 million "
            "cusecs on 14 August 2010. Breach of the right bank embankment near Sukkur "
            "inundated over 200,000 acres of cropland. More than 1.2 million residents were "
            "displaced across Sindh and moved to elevated highways and relief camps. "
            "The Pakistan Army and Navy conducted boat-based rescue operations across "
            "interior Sindh for 45 days. Livestock losses exceeded 700,000 animals. "
            "NDMA coordinated the delivery of 900,000 food packages, 250,000 tents, and "
            "medical supplies to affected families."
        ),
    },
    {
        "source": "NDMA Flood Report 2022",
        "title": "National Disaster Management Authority — 2022 Monsoon Floods: Situation Report No. 48",
        "content": (
            "The 2022 Pakistan monsoon floods were declared a national emergency on 25 August "
            "2022. Total affected population reached 33 million across all four provinces. "
            "Sindh was the worst-affected province with 7.9 million people displaced. "
            "Balochistan recorded flash floods destroying 73% of the road network. "
            "The Indus River system carried 35% above-average discharge throughout August. "
            "Manchar Lake in Sindh expanded to 1,500 km² — five times its normal size — "
            "threatening Sehwan and Dadu districts. Deliberate controlled breaching of "
            "embankments near Dadu was undertaken to protect the city of Hyderabad. "
            "NDMA, UN OCHA, and international partners launched a $160 million flash appeal. "
            "Total economic damage was estimated at $30 billion. Crop losses included "
            "43% of Pakistan's cotton production. The floods damaged or destroyed 1.7 million "
            "homes, with Sindh accounting for 1.2 million damaged housing units."
        ),
    },
    {
        "source": "FFD River Stage Report",
        "title": "Federal Flood Division — River Stage and Discharge Bulletin: Indus System 2024",
        "content": (
            "The Federal Flood Division (FFD) under the Ministry of Water Resources monitors "
            "river stages and discharge at 31 gauge stations across Pakistan. Key thresholds: "
            "NORMAL status is below 80% of mean annual flood; HIGH status is between 80% and "
            "120% of mean annual flood; EXTREME status exceeds 120% of mean annual flood. "
            "Tarbela Dam (Indus): current inflow 280,000 cusecs, outflow 220,000 cusecs, "
            "trend Rising. Chashma Barrage (Indus): inflow 340,000 cusecs, outflow 310,000 "
            "cusecs, status HIGH. Taunsa Barrage (Indus): inflow 420,000 cusecs, status HIGH. "
            "Guddu Barrage (Indus): inflow 480,000 cusecs, outflow 460,000 cusecs, status EXTREME. "
            "Sukkur Barrage (Indus): inflow 510,000 cusecs, status EXTREME. "
            "Trimmu Headworks (Chenab-Jhelum): inflow 185,000 cusecs, status HIGH. "
            "Nowshera Gauge (Kabul River): inflow 95,000 cusecs, status HIGH, trend Rising. "
            "Station operators are advised to report any embankment vulnerability to provincial "
            "irrigation departments immediately."
        ),
    },
    {
        "source": "FFD River Stage Report",
        "title": "Federal Flood Division — Historical Flood Peaks: Kabul River System",
        "content": (
            "The Kabul River originates in Afghanistan and enters Pakistan near Landi Kotal, "
            "Khyber Pakhtunkhwa. It flows through Peshawar, Charsadda, and Nowshera before "
            "joining the Indus near Attock. Historical peak discharges at Nowshera gauge: "
            "1929 — 850,000 cusecs (record flood); 1992 — 620,000 cusecs; 2010 — 850,000 "
            "cusecs (record matched). Charsadda lies in the floodplain of the Kabul River "
            "and is particularly vulnerable to rapid inundation when embankments fail. "
            "The river's steep gradient causes flash-flood conditions within 6–8 hours of "
            "heavy rainfall in the upper catchment. An early warning system with gauges at "
            "Warsak Dam and Nowshera provides a 4-hour lead time for downstream districts. "
            "The 2010 event at Charsadda resulted in the displacement of residents across "
            "the entire district, with the Kabul River overflow inundating the low-lying "
            "union councils of Tangi and Shabqadar completely."
        ),
    },
    {
        "source": "NDMA Preparedness Guidelines",
        "title": "NDMA Standard Operating Procedures: Flood Response and Evacuation",
        "content": (
            "NDMA flood response is activated in three phases. Phase 1 (Alert): Provincial "
            "Disaster Management Authorities (PDMAs) issue public warnings when river gauges "
            "reach HIGH status. District administrations pre-position rescue boats and relief "
            "supplies at designated staging areas. Phase 2 (Evacuation): Evacuation orders "
            "are issued for flood-prone union councils. The Pakistan Army and Civil Armed "
            "Forces conduct door-to-door evacuation in areas without road access. Vulnerable "
            "populations (elderly, women with infants, persons with disabilities) are given "
            "priority. Relief camps with minimum standards (SPHERE) including water, "
            "sanitation, shelter, and food are established at schools and public buildings. "
            "Phase 3 (Relief and Recovery): NDMA coordinates distribution of food parcels, "
            "tarpaulins, and non-food items. Cash grants of PKR 25,000 per household are "
            "issued through BISP Nadra verification. Agricultural input support (seeds, "
            "fertilizer) is provided in the post-flood Rabi season."
        ),
    },
    {
        "source": "Historical Flood Records Pakistan",
        "title": "Pakistan Meteorological Department — Historical Monsoon Flood Statistics 1950–2023",
        "content": (
            "Pakistan experiences monsoon flooding annually from July through September. "
            "Major flood events: 1950 (Punjab breaches), 1973 (10 million displaced), "
            "1976 (Indus basin), 1992 (KPK flash floods), 2010 (super floods — worst on "
            "record, 20 million displaced, $43 billion damage), 2011 (post-Sindh flooding), "
            "2022 (33 million affected, $30 billion damage). Swat River floods are "
            "triggered by glacial lake outburst floods (GLOFs) from the Karakoram glaciers "
            "combined with monsoon rainfall. Swat district recorded peak discharge at "
            "Chakdara gauge of 350,000 cusecs in 2010. Climate change projections indicate "
            "a 20% increase in extreme monsoon events by 2050 (IPCC AR6). The probability "
            "of a repeat 2010-scale flood has increased from a 1-in-100-year to 1-in-50-year "
            "event according to climate modeling by PCRWR. Districts most vulnerable to "
            "Indus flooding: Jacobabad, Larkana, Sukkur, Shikarpur, Kashmore, Rajanpur. "
            "Districts most vulnerable to KPK river flooding: Charsadda, Nowshera, Peshawar, "
            "Swat, Dir Lower."
        ),
    },
]
