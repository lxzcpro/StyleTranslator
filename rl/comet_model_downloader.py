from comet import download_model, load_from_checkpoint

model_path = download_model("Unbabel/wmt22-cometkiwi-da", "Z:\model")
print(model_path)
model = load_from_checkpoint(model_path)

data = [
    {
        "src": """Masterfully working across subject matter, Siso has generated a prolific series of landscapes, portraits, and still-life works rendered in either acrylic, pastel, pencil or watercolor. Drawing from family portraits, his own reference photographs, and recollection, his colorful compositions demonstrate his range of interests and skill across media. Siso's tropical landscapes and seascapes reflect the geographies of his past, employing rich patterns and incorporating people to make meaningful connections between culture, memory and the environment. Siso titles his artworks in a mix of Spanish and English, signifying the celebrated and integral complexities of his life in Los Angeles County. "Vicente Siso: Memories of the Land and Water" opens on Saturday, Jan. 13, with a reception from 6-8 p.m. The exhibition is on view through Sunday, March 3.""",
        "mt": """西索（Siso）在不同题材间游刃有余，创作了大量风景、肖像和静物作品，采用丙烯、粉彩、铅笔或水彩等媒介。他从家庭肖像、自己的参考照片以及回忆中汲取灵感，色彩斑斓的作品展现了他在不同媒介上的广泛兴趣和高超技艺。西索的热带风景和海景作品反映了他过往生活的地理环境，运用丰富的图案并融入人物，以建立文化、记忆与环境之间的意义联系。西索的作品标题采用西班牙语和英语混合的形式，彰显了他在洛杉矶县生活中的复杂性和独特性。“维森特·西索：土地与水的记忆”将于 1 月 13 日（星期六）开幕，当晚 6 点至 8 点举行开幕式。展览将持续至 3 月 3 日（星期日）。"""
    },
    {
        "src": """Masterfully working across subject matter, Siso has generated a prolific series of landscapes, portraits, and still-life works rendered in either acrylic, pastel, pencil or watercolor. Drawing from family portraits, his own reference photographs, and recollection, his colorful compositions demonstrate his range of interests and skill across media. Siso's tropical landscapes and seascapes reflect the geographies of his past, employing rich patterns and incorporating people to make meaningful connections between culture, memory and the environment. Siso titles his artworks in a mix of Spanish and English, signifying the celebrated and integral complexities of his life in Los Angeles County. "Vicente Siso: Memories of the Land and Water" opens on Saturday, Jan. 13, with a reception from 6-8 p.m. The exhibition is on view through Sunday, March 3.""",
        "mt": """妙趣横生地跨越不同题材，Siso 创作了一系列丰富多彩的风景画、肖像画与静物作品，媒介包括丙烯、粉彩、铅笔和水彩。他从家族肖像、自身参考照片以及记忆中汲取灵感，用色彩斑斓的构图展现了他在不同媒介上的兴趣与技艺。Siso 的热带风景与海景作品折射出他过去的地理背景，通过丰富的图案和人物的融入，将文化、记忆与环境紧密联系起来。Siso 用西班牙语与英语混合为作品命名，彰显了他在洛杉矶县生活中所经历的复杂而多元的人生。展览《Vicente Siso: Memories of the Land and Water》将于1月13日（周六）开幕，开幕酒会时间为晚上6点至8点，展览将持续至3月3日（周日）。
"""
    },
    {
        "src": """Masterfully working across subject matter, Siso has generated a prolific series of landscapes, portraits, and still-life works rendered in either acrylic, pastel, pencil or watercolor. Drawing from family portraits, his own reference photographs, and recollection, his colorful compositions demonstrate his range of interests and skill across media. Siso's tropical landscapes and seascapes reflect the geographies of his past, employing rich patterns and incorporating people to make meaningful connections between culture, memory and the environment. Siso titles his artworks in a mix of Spanish and English, signifying the celebrated and integral complexities of his life in Los Angeles County. "Vicente Siso: Memories of the Land and Water" opens on Saturday, Jan. 13, with a reception from 6-8 p.m. The exhibition is on view through Sunday, March 3.""",
        "mt": """在中世纪时期的宏伟工程中，斯吉奥斯从未间断，在绘画、肖像画和夜景画方面创造了多种多样的形式。参考他的亲兄弟们、自身的照片和回忆，他色彩明亮的作品展示了他在多种媒体上的广博兴趣和技巧。斯吉奥斯的热带风光和水上景观反映了他在过去的时光中发展的伟大，充分利用了丰富的图案并融入了人们之间的具体联系，以此文化和记忆与环境和谐交融。斯吉奥斯在他的纪念碑上镌刻了中、英文，并且以他充满争议的艺术作品致谢公众。"维克森·斯吉奥斯:洛杉矶的山脉和水"将于2021年1月13日于洛杉矶县格兰特中心举行，开幕时间为下午6点，整个展览将持续到3月3日。"""
    }
]
model_output = model.predict(data, batch_size=8, gpus=1)
print (model_output)
