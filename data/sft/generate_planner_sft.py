import json
import os

data = [
    {
        "input": "给我放一首周杰伦的《稻香》",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "稻香",
                "artist_name": "周杰伦"
            },
            "context": "播放周杰伦收录于《魔杰座》的代表作稻香",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "用户指定了确切的歌曲名并要求播放，不需要进行推荐检索。"
        }
    },
    {
        "input": "我想听《Hotel California》",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "Hotel California"
            },
            "context": "播放加州旅馆",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "指定了具体英文外文歌曲名《Hotel California》，直接触发播放。"
        }
    },
    {
        "input": "播放昨天排行榜第一名的歌",
        "output": {
            "intent_type": "search",
            "parameters": {},
            "context": "查询昨天音乐排行榜的冠军歌曲并播放",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "昨天 音乐排行榜 第一名 冠军歌曲"
            },
            "reasoning": "用户未给出具体歌名，'昨天排行榜第一名'需要时效性联网搜索才能确定具体曲目，属于search意图。"
        }
    },
    {
        "input": "来一首《Shape of You》吧，黄老板的",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "Shape of You",
                "artist_name": "Ed Sheeran"
            },
            "context": "播放 Ed Sheeran 的 Shape of You",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "明确歌名及歌手简称要求播放特定歌曲。"
        }
    },
    {
        "input": "林俊杰有哪些好听的歌",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "林俊杰"
            },
            "context": "搜索林俊杰的歌曲作品",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "林俊杰",
                    "JJ Lin"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "用户想要查询特定歌手的歌曲，需要从图谱获取该歌手实体信息。加上vector做兜底兜住'好听'这种模糊情感。"
        }
    },
    {
        "input": "搜一下 Coldplay 的经典曲目",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "Coldplay"
            },
            "context": "搜索酷玩乐队的经典作品",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "Coldplay",
                    "酷玩乐队"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "明确指定外文乐队名，要求提供双语对照避免找不到实体。"
        }
    },
    {
        "input": "有没有五月天比较燃的歌？",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "五月天",
                "mood": "热血"
            },
            "context": "搜索五月天的热血歌曲",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "五月天",
                    "Mayday"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "热血",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "复合查询，既包含歌手实体，又包含'热血'这种情绪过滤，还需要向量抓取'燃'的主观声学特征。"
        }
    },
    {
        "input": "我想听 Taylor Swift 之前出的乡村音乐",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "Taylor Swift",
                "genre": "乡村乐"
            },
            "context": "搜索泰勒斯威夫特的乡村音乐风格作品",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "Taylor Swift",
                    "泰勒·斯威夫特"
                ],
                "graph_genre_filter": "country",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "指定了歌手与流派，将流派映射到规范的 country 字段。"
        }
    },
    {
        "input": "刚刚那首歌怎么下载",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {},
            "context": "询问刚刚推荐曲目的下载方式",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "纯功能性请求，无需推荐检索。"
        }
    },
    {
        "input": "把这几首都加入我的歌单，发个下载链接",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {},
            "context": "用户请求获取当前推荐的全部曲目并下载",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "确认获取歌曲的操作，不涉及内容检索。"
        }
    },
    {
        "input": "想要第二首歌的无损音源",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {},
            "context": "获取选定歌曲的无损版本",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "属于下载/获取资源的行为意图。"
        }
    },
    {
        "input": "怎么保存到本地啊",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {},
            "context": "询问歌曲本地化存储方法",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "关于保存/获取方式的问题。"
        }
    },
    {
        "input": "放点我以前收藏过的音乐",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "播放用户收藏列表",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "明确指向用户历史偏好与收藏目录的调用。"
        }
    },
    {
        "input": "昨天我点赞过的一首粤语歌是啥",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "查找用户最近点赞的记录",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "查询自己的个人点赞数据，不需要走外网/知识图谱检索推荐。"
        }
    },
    {
        "input": "我想重温一下最喜欢的那些老歌",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "回顾用户红心歌单",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "边界情况，'最喜欢的'即用户红心收藏，优先查收藏夹。"
        }
    },
    {
        "input": "把你上次给我推荐过的觉得不错的再发一遍",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "调取最近历史交互获取喜欢的推荐",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "回顾过往历史操作，本质是返回用户历史数据。"
        }
    },
    {
        "input": "有没有类似陈奕迅风格的歌",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "陈奕迅"
            },
            "context": "寻找与陈奕迅风格相似的歌曲",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "陈奕迅",
                    "Eason Chan"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "通过给定歌手来推荐风格相似的音乐，需要图谱查找歌手属性，同时依赖向量搜索声学相似度。"
        }
    },
    {
        "input": "推荐几首像 Justin Bieber 那种流行风格的",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "Justin Bieber",
                "genre": "流行"
            },
            "context": "推荐类似贾斯汀比伯的流行乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "Justin Bieber",
                    "贾斯汀·比伯"
                ],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "明确提出相似艺人和流派风格，综合利用图谱流派过滤和向量声学查找。"
        }
    },
    {
        "input": "听腻了华晨宇，换点跟他差不多感觉的别的歌手的歌",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "华晨宇"
            },
            "context": "推荐风格类似于华晨宇的其他歌手作品",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "华晨宇",
                    "Hua Chenyu"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": "Chinese",
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "要求查找相似歌手，用图谱限制华语区域或依赖向量计算声学相似度。"
        }
    },
    {
        "input": "我比较喜欢 The Beatles，给我推点符合我口味的",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "The Beatles"
            },
            "context": "基于甲壳虫乐队进行相似推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "The Beatles",
                    "披头士乐队",
                    "甲壳虫乐队"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "利用具体乐队做种子进行推荐扩展。"
        }
    },
    {
        "input": "照着米津玄师的那个感觉，来几首日文歌",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "米津玄师",
                "language": "日语"
            },
            "context": "推荐类似米津玄师风格的日语歌",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "米津玄师",
                    "Kenshi Yonezu"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": "Japanese",
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "包含具体的实体和语言过滤限制。"
        }
    },
    {
        "input": "推荐摇滚乐",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "摇滚"
            },
            "context": "简单推荐摇滚流派音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "rock",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "非常直接明确的流派要求。图谱限制摇滚，向量作声学补充。"
        }
    },
    {
        "input": "来点重金属，要国内的乐队",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "重金属",
                "region": "国内"
            },
            "context": "推荐大陆重金属风格音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "metal",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": "Mainland China",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "不仅指定流派，还限定了地区，映射地域过滤器。"
        }
    },
    {
        "input": "想听一些古典音乐陶冶一下情操",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "古典"
            },
            "context": "推荐古典流派音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "classical",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "古典乐是明确流派。主观表达'陶冶情操'也适合让向量进行一定捕捉。"
        }
    },
    {
        "input": "推荐一些 chill 的 lo-fi beats",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "lo-fi/电子",
                "vibe": "chill"
            },
            "context": "推荐放松氛围的低保真电子节奏",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "electronic",
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "中英混杂口语，lo-fi归为electronic流派，chill归为放松情绪。严重依赖向量找氛围。"
        }
    },
    {
        "input": "给我整点老派嘻哈和 R&B",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "嘻哈/R&B"
            },
            "context": "怀旧老派说唱与节奏布鲁斯",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "hip-hop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "怀旧",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "多个流派提及，可选主导流派hip-hop，通过向量处理'老派'复古感，并加入怀旧mood。"
        }
    },
    {
        "input": "你觉得周杰伦最伟大的一首歌是哪首",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "与用户闲聊对周杰伦作品的主观评价",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "这是一个主观探讨问题，并非让系统播放或推荐特定歌曲。"
        }
    },
    {
        "input": "你平时怎么选歌的",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "用户询问系统的推荐机制",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "提问探讨性质，不触发检索。"
        }
    },
    {
        "input": "早上好啊",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "日常打招呼问候",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "单纯问候寒暄。"
        }
    },
    {
        "input": "我不喜欢上一首",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "用户对前一首歌给出负面反馈",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "反馈意见并不是新的推荐查询，Planner应分配给对话链路处理更新短期偏好即可。"
        }
    },
    {
        "input": "失恋了心好痛，有没有能哭的歌",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "伤感/哭泣"
            },
            "context": "寻找能释放失恋情绪的悲伤歌曲",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "伤感",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "典型的隐含悲伤情绪的模糊意图。必须启用向量捕捉深度的情绪音乐氛围。"
        }
    },
    {
        "input": "我好累",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "疲惫/需要放松"
            },
            "context": "用户感到疲惫需要音乐舒缓放松",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "易错边界case：隐含情绪而非纯闲聊。需要转化为找放松音乐的推荐请求。"
        }
    },
    {
        "input": "今天心情超好！推点嗨的",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "开心/激动"
            },
            "context": "推荐使人快乐激动的高昂曲目",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "开心",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "直接体现好心情情绪，图谱过滤开心，向量补充嗨的听感。"
        }
    },
    {
        "input": "想要那种一个人在深夜里特别孤独的氛围感",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "孤独"
            },
            "context": "推荐体现深夜孤独氛围的氛围音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "孤独",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "极强的情感诉求，高度依赖向量进行氛围共鸣检索。"
        }
    },
    {
        "input": "来首能让我平静下来的轻音乐",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "平静"
            },
            "context": "推荐能平复心情的安静音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "平静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "纯情绪描述，不涉及歌手等。"
        }
    },
    {
        "input": "好气啊！！我要爆炸了",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "愤怒/暴躁"
            },
            "context": "用户处于极度愤怒状态，需要合适的发泄音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "愤怒",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "虽然未明确说点歌，但作为音乐助手，转化为以情绪推荐应对愤怒的发泄需求。"
        }
    },
    {
        "input": "找一些火热的歌",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "激情"
            },
            "context": "推荐目前流行氛围激燃火爆的歌曲",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "激情",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "注意这不是play_specific_song，而是基于热烈风格的泛推荐。"
        }
    },
    {
        "input": "有点想老婆了",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "浪漫/忧愁"
            },
            "context": "想念爱人，推荐浪漫或抒发忧郁的音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "浪漫",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "隐含意图，识别为思念浪漫氛围的推荐。"
        }
    },
    {
        "input": "感觉世界只剩我一个人，好致郁",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "忧郁/孤独"
            },
            "context": "推荐深沉致郁的孤独氛围音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "忧郁",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "强烈的感受词，依赖向量寻找听感贴合的声学表达。"
        }
    },
    {
        "input": "听罢有种恍若隔世的感觉",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "梦幻/空灵"
            },
            "context": "寻找具有梦幻空灵听感的超现实音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "梦幻",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "非常主观的听感描述，图谱只能映射大致情绪，必须用向量弥补。"
        }
    },
    {
        "input": "适合下雨天听的爵士，要慵懒一点",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "下雨天",
                "genre": "爵士",
                "vibe": "慵懒"
            },
            "context": "推荐适合雨天氛围的慵懒爵士乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "jazz",
                "graph_scenario_filter": "下雨天",
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "复合查询，有场景（下雨天）、流派（jazz）及主观声学（慵懒）。"
        }
    },
    {
        "input": "一个人开车在空旷的公路上，窗外风景飞驰而过",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "开车"
            },
            "context": "推荐驾驶场景下的公路音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "开车",
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "极具画面感的自然语言输入。场景解析为'开车'，声学向量可以充分消化空旷公路这种意境。"
        }
    },
    {
        "input": "快给我一点打鸡血的动感节奏，我要去跑步了",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "跑步",
                "mood": "热血"
            },
            "context": "推荐节奏强烈的跑步激励音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "跑步",
                "graph_mood_filter": "热血",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "场景是跑步，情绪可映射为热血。需要向量匹配动感节奏。"
        }
    },
    {
        "input": "今晚朋友聚会，放点有逼格的背景音",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "聚会"
            },
            "context": "推荐聚会场景的背景音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "聚会",
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "场景映射为'聚会'，'有逼格'是很主观的表达，依赖向量网络推导。"
        }
    },
    {
        "input": "早上通勤不想听太吵的，轻快就可以",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "通勤",
                "mood": "放松"
            },
            "context": "推荐早晨通勤时的轻快放松音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "通勤",
                "graph_mood_filter": "放松",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "场景映射为'通勤'，描述要求不吵且轻快，适合图谱作场景过滤和向量声学过滤。"
        }
    },
    {
        "input": "要准备明天考试了，搞点专注学习的bgm",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "学习"
            },
            "context": "推荐能帮助集中注意力的学习场景白噪音或纯音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "学习",
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "场景词为'学习'，明确指向activity。"
        }
    },
    {
        "input": "马上睡觉了，推几首助眠的粤语老歌",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "睡前",
                "language": "粤语",
                "mood": "怀旧"
            },
            "context": "推荐睡前播放的怀旧粤语助眠音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "睡前",
                "graph_mood_filter": "怀旧",
                "graph_language_filter": "Cantonese",
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "含有语言、场景，同时老歌隐含怀旧情绪。"
        }
    },
    {
        "input": "做饭切菜需要点节奏感",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "做饭"
            },
            "context": "推荐适合做饭场景且有节奏感的音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "做饭",
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "识别出特定的日常活动做饭作为场景。"
        }
    },
    {
        "input": "我正在打游戏练级，来几首战歌",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "打游戏",
                "mood": "激情"
            },
            "context": "推荐适合打游戏的激昂战歌",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "打游戏",
                "graph_mood_filter": "激情",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "打游戏场景，战歌具有极大强度的情感（激情/热血），结合向量进行查找。"
        }
    },
    {
        "input": "开启冥想时刻，放空思绪",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "冥想",
                "mood": "平静"
            },
            "context": "推荐适合深度冥想放松的灵性音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "冥想",
                "graph_mood_filter": "平静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "识别出冥想场景，并且用平静的情绪匹配图谱过滤。"
        }
    },
    {
        "input": "给我放陈粒的《易燃易爆炸》",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "易燃易爆炸",
                "artist_name": "陈粒"
            },
            "context": "播放陈粒演唱的具体歌曲易燃易爆炸",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "指定了具体华语女歌手及其代表歌曲名，属于纯点歌播放意图不用进行搜索推荐。"
        }
    },
    {
        "input": "播放 The Beatles 的 Hey Jude",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "Hey Jude",
                "artist_name": "The Beatles"
            },
            "context": "播放披头士的经典名曲Hey Jude",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "明确指定了外文乐队和具体英文歌曲名进行点播。"
        }
    },
    {
        "input": "我想听《反方向的钟》",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "反方向的钟"
            },
            "context": "播放用户指定的歌曲反方向的钟",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "带有书名号的纯歌曲名点播请求。"
        }
    },
    {
        "input": "来首 Billie Jean 听听",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "Billie Jean"
            },
            "context": "想要播放Billie Jean这首歌",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "明确说明要听某个特定名字的歌曲，不管带不带书名号都是 play 意图。"
        }
    },
    {
        "input": "切一首薛之谦的《认真的雪》",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "认真的雪",
                "artist_name": "薛之谦"
            },
            "context": "切歌并播放薛之谦的认真的雪",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "切换操作加明确的点歌指令。"
        }
    },
    {
        "input": "查一下邓紫棋最近一年发了什么新歌没",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "邓紫棋"
            },
            "context": "查找邓紫棋近期发行的最新单曲",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "邓紫棋",
                    "G.E.M."
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "邓紫棋 最近一年 新歌 2025"
            },
            "reasoning": "搜索具体歌手的最新歌曲动态，启用图谱确认歌手实体，同时因为有时效性（最近一年新歌）必须开启 Web 搜索。"
        }
    },
    {
        "input": "帮我搜索一下有没有比较激烈的 Linkin Park 的摇滚",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "Linkin Park",
                "genre": "摇滚",
                "mood": "热血"
            },
            "context": "搜索林肯公园比较热血激烈的摇滚乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "Linkin Park",
                    "林肯公园"
                ],
                "graph_genre_filter": "rock",
                "graph_scenario_filter": None,
                "graph_mood_filter": "热血",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "指定了乐队名、流派和激烈主观情绪倾向，必须图谱与向量搜索同时上阵抓取硬核歌曲。"
        }
    },
    {
        "input": "李荣浩好听的抒情歌有哪些",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "李荣浩"
            },
            "context": "搜索李荣浩演唱的好听的抒情风格歌曲",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "李荣浩",
                    "Li Ronghao"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "浪漫",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "特定歌手搜索，‘抒情’作为主观听感倾向可以映射到浪漫，同时靠向量捕捉煽情氛围。"
        }
    },
    {
        "input": "找一找那个叫 Adele 的女歌手的歌",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "Adele"
            },
            "context": "通过名字搜索歌手阿黛尔的所有作品",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "Adele",
                    "阿黛尔"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "寻找特定歌手实体，单纯的图谱搜索任务。"
        }
    },
    {
        "input": "搜个周深的国风歌曲",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "周深",
                "genre": "国风"
            },
            "context": "搜索周深演唱的中国风风格音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "周深",
                    "Zhou Shen"
                ],
                "graph_genre_filter": "folk",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": "Chinese",
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "除了歌手之外强调了国风，国风可近似映射到民谣分类结合中国语言，再加上向量的传统乐器声学捕捉。"
        }
    },
    {
        "input": "张学友的粤语老歌",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "张学友",
                "language": "粤语"
            },
            "context": "搜索张学友演唱的粤语怀旧金曲",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "张学友",
                    "Jacky Cheung"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "怀旧",
                "graph_language_filter": "Cantonese",
                "graph_region_filter": "Hong Kong",
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "带有限制了特定歌手、粤语语言和怀旧年代感的明确图谱搜索指令。"
        }
    },
    {
        "input": "刚刚这首帮我存一下",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "这首歌"
                ]
            },
            "context": "用户请求保存刚刚播放的音乐",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "对于音乐操作处理工具的显式保存指令。"
        }
    },
    {
        "input": "打包下载昨晚听的那个歌单",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "昨晚播放的歌单内的歌曲"
                ]
            },
            "context": "用户想要把昨晚的历史推歌全部下载回来",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "这并不是推荐而是将之前的音频源文件进行获取请求。"
        }
    },
    {
        "input": "这些歌怎么添加到我的本地库里",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "前面检索出的歌曲"
                ]
            },
            "context": "寻求将推荐曲目导入本地音乐库的路径",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "表达对系统功能中下载/保存的意图呼唤。"
        }
    },
    {
        "input": "好听，下载！",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "当前播放曲目"
                ]
            },
            "context": "表示赞同并立刻要求下载目前的音乐文件",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "即使带有好听等评价语气，终极意图还是触发了下载动作。"
        }
    },
    {
        "input": "能把这首歌的 MP3 发给我吗",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "当前播放歌曲"
                ]
            },
            "context": "请求获取这首歌源格式文件发送给自己",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "请求明确指定了获取MP3也就是要求下载源文件动作。"
        }
    },
    {
        "input": "觉得好焦虑啊，有什么能让人安静下来的音乐吗",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "平静"
            },
            "context": "用户表达内心焦虑，希望能找到平复心情的安静音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "平静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "表述中的“焦虑”需要被转化为正向疗愈的“平静”图谱情绪。开启向量用以寻找柔和安静的声音空间。"
        }
    },
    {
        "input": "工作被老板骂了，好想哭",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "伤感"
            },
            "context": "遭遇职场挫折，通过悲伤释放情绪",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "工作",
                "graph_mood_filter": "伤感",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "边界 case：极度隐式的表达。虽然提到了工作，但主要是想哭（情绪主导），通过向量捕获情感悲伤氛围。"
        }
    },
    {
        "input": "今天拿到offer了爽死了！！来点应景的",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "激情"
            },
            "context": "拿到offer非常开心振奋，需要激昂的音乐反馈喜悦",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "激情",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "非常亢奋的情绪意图表达，利用向量找到听感炸裂的曲子。"
        }
    },
    {
        "input": "这阴郁的天气搞得我心情好沉重",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "忧郁"
            },
            "context": "天气导致心情压抑忧郁，推荐对应氛围或者治愈的歌曲",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "忧郁",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "情绪主导的需求。图谱过滤器设定为忧郁，利用向量找出共鸣音色。"
        }
    },
    {
        "input": "想谈恋爱啦，给点粉红泡泡感觉的歌",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "浪漫"
            },
            "context": "用户由于想恋爱而希望能感受到浪漫的粉色氛围",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "浪漫",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "恋爱期待对应的就是浪漫治愈系风格。图谱选用浪漫情绪来筛掉其他不合适的风格。"
        }
    },
    {
        "input": "晚上一个人空虚寂寞冷",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "孤独"
            },
            "context": "在夜晚感到强烈的孤独感，推荐深度共鸣曲目",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "孤独",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "隐喻极强的寂寞独处场景，本质是纯情绪发泄，过滤孤独情绪。"
        }
    },
    {
        "input": "推荐一些炸裂的核金属",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "核金属"
            },
            "context": "寻找具有极端听感的金属核流派音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "metal",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "具体的细分音乐类别需求。将核金属归属到泛 metal 流派图谱大类，用向量做'炸裂'的特征提取补救。"
        }
    },
    {
        "input": "喜欢纯粹的 Jazz 乐，有没有好点的单曲",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "爵士乐"
            },
            "context": "向用户推荐纯粹的爵士流派优质音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "jazz",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "中英混合指代流派 jazz，且无具体人物出现，是典型的指定流派探索。"
        }
    },
    {
        "input": "有没有那种古巴风情味道的拉美音乐",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "拉丁",
                "region": "拉美地区"
            },
            "context": "请求带有古巴风格元素的拉丁地区特定音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "latin",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": "Western",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "非常针对性的流派（拉丁）加上特殊氛围诉求。向量必须上场负责拉拢那些带有特定'风情'声学质感的混音作品。"
        }
    },
    {
        "input": "来首 R&B，最好是带点忧郁色调的",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "R&B",
                "mood": "忧郁"
            },
            "context": "推荐同时具备节奏布鲁斯曲风和忧郁情绪的曲子",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "r&b",
                "graph_scenario_filter": None,
                "graph_mood_filter": "忧郁",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "完美复合查询，既包含流派 R&B 也囊括明确情绪忧伤，在保证类型对味同时能贴近失落情感。"
        }
    },
    {
        "input": "推点 Indie 独立音乐",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "独立"
            },
            "context": "寻找具有较高艺术个性的小众独立类型音乐推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "indie",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "用户清晰点出 Indie 这个标签，这通常具备多变不羁的艺术化听觉风格，因此靠图谱加向量实现最佳兜底发现。"
        }
    },
    {
        "input": "想听国内的乡村民谣",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "民谣",
                "region": "内地"
            },
            "context": "推荐大陆地区中国语言演唱的村屯民谣或城市民谣风格歌曲",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "folk",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": "Chinese",
                "graph_region_filter": "Mainland China",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "指定流派民谣加国内地区限制词，因此可以对图谱使用双过滤同时开启地区和类型以缩小检索域。"
        }
    },
    {
        "input": "给我放几首像邓紫棋那样铁肺狂飙高音的歌手",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "邓紫棋",
                "mood": "激情"
            },
            "context": "针对极高演唱能力和飙高音特质，寻找与邓紫棋同类型的歌手歌曲",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "邓紫棋",
                    "G.E.M."
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "激情",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "并不想单纯听她这一个人的歌，而是需要同属性对标。高音声压强是一种严重依赖物理声学特质和向量匹配的强音频特征。"
        }
    },
    {
        "input": "我不怎么认识英文歌手，但我就想要之前那种听 Adele 时心碎的感觉",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "Adele",
                "mood": "伤感"
            },
            "context": "推荐具备伤感心碎底色，类似阿黛尔演唱特质的作品",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "Adele",
                    "阿黛尔"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "伤感",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "用户并不明确追求这一个实体，而是以该实体作为一种体验象征符号发起发散性的音色和心境配对需求查找。"
        }
    },
    {
        "input": "有没有日韩乐队跟五月天天团那种风格相近的",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "五月天",
                "region": "日韩"
            },
            "context": "跨国找寻精神内核/类型类似于五月天的日韩两地音乐组合",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "五月天",
                    "Mayday"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": "Japan",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "给出实体做跳板，同时限定了地区大板块。要求通过该实体衍生出同样流行摇滚质感的新音乐源头。"
        }
    },
    {
        "input": "找一找和 BLACKPINK 曲风差不多的女团作品",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "BLACKPINK",
                "genre": "流行"
            },
            "context": "搜罗符合韩国K-Pop高工业化酷炫劲爆女团质感的新作品",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "BLACKPINK"
                ],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": "Korean",
                "graph_region_filter": "Korea",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "非常鲜明的基于知名偶像团体的同质化推荐延展意图操作。"
        }
    },
    {
        "input": "喜欢毛不易的叙事感，给我推点别的",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "毛不易"
            },
            "context": "基于毛不易以词带情的特殊说唱民谣式音乐氛围进行扩展",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "毛不易",
                    "Mao Buyi"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "提取了特定歌手的核心长处并依此索取非他本人的全新音乐单曲做推荐。"
        }
    },
    {
        "input": "类似酷玩乐队，来点",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "酷玩乐队"
            },
            "context": "寻找和酷玩乐队风格相类似的歌曲组合推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "酷玩乐队",
                    "Coldplay"
                ],
                "graph_genre_filter": "rock",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "给出中译著名乐团名字要求类似替代品发掘。"
        }
    },
    {
        "input": "在我的喜爱列表里随机抽几首",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "从用户已标星或者确认过有喜好倾向的私域数据库中抽选并播放",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "这是指向用户明确保存红心喜好的个人私密知识库调用过程。"
        }
    },
    {
        "input": "昨天那几首挺棒的，你再播一次之前我听嗨过的",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "回顾近期被标注为极高满意度的历史推荐列表",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "重点提取用户自身历史数据里的正面回馈行为作为选曲源头。"
        }
    },
    {
        "input": "把我点赞的拿出来放",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "直接提取出红心点赞歌单中的项目进行反馈播放",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "非常明确的具体产品级互动意图：我点赞过的歌单播放功能激活。"
        }
    },
    {
        "input": "推荐一首我以前常听的",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "以个人的播放次数或者长历史权重做打底来播放情怀存货",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "基于本人历史长期统计记录的访问诉求。"
        }
    },
    {
        "input": "查查我的收藏列表",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "通过浏览查看用户的收藏记录进行交互响应返回数据内容",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "涉及对自有数据的检阅请求指令无需外连大网和全局检索库操作。"
        }
    },
    {
        "input": "现在得专注写代码工作了，不想被打扰",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "工作"
            },
            "context": "正在推进写代码的高专注脑力消耗工作场景",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "工作",
                "graph_mood_filter": "平静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "明确的写代码/工作学习高度专注场景，不适合太高干扰性音乐所以需要向量协助把关静音、清透质感。"
        }
    },
    {
        "input": "准备开车自驾游，求提神醒脑的公路音乐",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "开车",
                "mood": "热血"
            },
            "context": "适合公路旅行高速驾驶时振作精神避免犯困的节奏感配乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "开车",
                "graph_mood_filter": "激情",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "开车场景外挂需要提神的情绪，因此开启对应的双向要求限制和兜底检测。"
        }
    },
    {
        "input": "在地铁通勤人挤人太烦了，隔绝噪音的",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "通勤"
            },
            "context": "需要为烦人的大量人类拥挤通勤场景提供心理减负的包裹感音波",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "通勤",
                "graph_mood_filter": "治愈",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "虽然伴随着负极性情绪烦闷，但这实际属于通勤路途场景下的防打扰治愈型要求。靠向量搜集高掩蔽效用的声音。"
        }
    },
    {
        "input": "约会！跟女朋友一起烛光晚餐！",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "约会",
                "mood": "浪漫"
            },
            "context": "为高私密感、高质量的浪漫晚餐约会时光加成感情配乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "约会",
                "graph_mood_filter": "浪漫",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "极强的约会气氛组布置需求。图谱直接卡位约会标签、情感锚定浪漫基底。"
        }
    },
    {
        "input": "做饭切菜时候听点欢快的",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "做饭",
                "mood": "开心"
            },
            "context": "居家在厨房切菜做饭做家务时能提高干劲不觉得枯燥的欢乐鼓点节奏",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "做饭",
                "graph_mood_filter": "开心",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "典型的家务活操作附加开朗愉快心境双维度锁定目标靶心的意图分类。"
        }
    },
    {
        "input": "下雨天坐在窗边看书",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "看书"
            },
            "context": "结合了由于外部下雨天气以及自身在家安安静静看书读书的休闲阅读综合体画面感",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "看书",
                "graph_mood_filter": "平静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "看书本身就是高度宁静内向的行为再加上下雨天更是加深了chill感。靠向量把控白噪音和音乐的均衡不干扰书卷气。"
        }
    },
    {
        "input": "为什么周杰伦那么长时间不出专辑了",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "对于顶流偶像拖延不出新专进行普通粉丝八卦交流",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "并非想要推荐他任何当前或历史歌曲的查阅需求，仅仅开启一个行业或者偶像八卦讨论。"
        }
    },
    {
        "input": "这破歌太难听了，快切掉",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "收到严厉拒绝与反感的当前推荐播放项目负向反馈意见",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "这是一个对上一操作直接否决的强意愿打断指令而非正向搜寻。交给对话管理层进行反转安抚并更新图谱厌恶值。"
        }
    },
    {
        "input": "你懂什么是真正的爱情吗",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "与智能体AI进入了纯形而上的哲学发问或生活感悟扯皮阶段",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "完全无关乎放歌的瞎聊天打屁。"
        }
    },
    {
        "input": "哈哈这首不错我挺满意的",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "用户发出的短促正向喜悦体验好评。并无进一步查询诉求",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "虽然夸了不错但也仅仅是对之前动作的结果做认可不需额外搜索出新单子。转入常态聊天接话茬记录档案。"
        }
    },
    {
        "input": "你们这个软件是怎么打分的？",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "对于当前项目或者应用背后机理和使用方式的操作盘问及FAQ引导",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "关于app怎么用的咨询，根本扯不到要放什么或者抓取啥子。由大语言模型直接回复解答机制原理即可。"
        }
    },
    {
        "input": "那给我放林俊杰的《江南》",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "江南",
                "artist_name": "林俊杰"
            },
            "context": "由于明确歌名，直接播放林俊杰的江南",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "用户指定了具体歌手与代表作，为纯点歌意图。"
        }
    },
    {
        "input": "能放一遍《Rolling in the Deep》吗",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "Rolling in the Deep"
            },
            "context": "播放具体英文歌曲Rolling in the Deep",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "不带歌手名的明确英文歌曲名点播。"
        }
    },
    {
        "input": "想听周杰伦的那首晴天",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "晴天",
                "artist_name": "周杰伦"
            },
            "context": "希望听周杰伦创作的晴天",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "未借助书名号的具体曲目表达，依旧是点播。"
        }
    },
    {
        "input": "马上播《起风了》",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "起风了"
            },
            "context": "强烈的要求立即播放起风了",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "明确书名号点播，无歌手前置。"
        }
    },
    {
        "input": "我要听蔡依林的日不落",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "日不落",
                "artist_name": "蔡依林"
            },
            "context": "用户发起了具体的歌曲日不落的点播",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "点歌场景，无需推荐和网络检索。"
        }
    },
    {
        "input": "播放朴树的《平凡之路》",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "平凡之路",
                "artist_name": "朴树"
            },
            "context": "播放朴树经典作平凡之路",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "典型的人名+歌名格式的点播播放。"
        }
    },
    {
        "input": "张国荣有哪些经典的粤语歌",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "张国荣",
                "language": "粤语"
            },
            "context": "搜索哥哥张国荣经典的粤语音乐作品",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "张国荣",
                    "Leslie Cheung"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": "Cantonese",
                "graph_region_filter": "Hong Kong",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "指定了歌手与语言倾向，需要图谱匹配歌手实体和粤语语种，“经典”可用向量辅助。"
        }
    },
    {
        "input": "有没有蔡健雅稍微伤感一些的歌",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "蔡健雅",
                "mood": "伤感"
            },
            "context": "寻找蔡健雅充满伤感悲伤情绪的曲子",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "蔡健雅",
                    "Tanya Chua"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "伤感",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "搜索具体艺人但附加明显情绪要求，图谱中填入伤感情绪词并由向量发力捕捉落寞感。"
        }
    },
    {
        "input": "给我搜几首 Michael Jackson 的舞曲",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "Michael Jackson",
                "genre": "流行舞曲"
            },
            "context": "搜寻迈克尔杰克逊的快节奏流行舞曲",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "Michael Jackson",
                    "迈克尔·杰克逊"
                ],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "查询国外特定歌手同时用“舞曲”指代了动感流派，双语实体必须包含。"
        }
    },
    {
        "input": "陈奕迅最近三年出过专辑吗，都有啥",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "陈奕迅"
            },
            "context": "查询陈奕迅近期是否发布了新专辑",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "陈奕迅",
                    "Eason Chan"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "陈奕迅 最近三年 新专辑"
            },
            "reasoning": "对于歌手近况及时效性非常强的盘点，必须结合联网引擎搜索弥补本地库的时滞。"
        }
    },
    {
        "input": "帮我找找万能青年旅店那些纯乐器演奏的曲子",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "万能青年旅店"
            },
            "context": "寻找万能青年旅店创作的无人声纯器乐作品",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "万能青年旅店",
                    "Omnipotent Youth Society"
                ],
                "graph_genre_filter": "indie",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "查询独立乐队，但对听感提出极高要求“纯器乐”，必须依赖下游声学向量精准筛选。"
        }
    },
    {
        "input": "听听王菲早期的金曲",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "王菲"
            },
            "context": "搜索并播放王菲早年广为流传的经典作品",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "王菲",
                    "Faye Wong"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "怀旧",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "“早期”、“金曲”都带有强烈的年代记忆与情感印记，图谱可标注怀旧，结合向量识别母带质感。"
        }
    },
    {
        "input": "把上面你说的那三首歌发给我",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "推荐列表中的三首歌曲"
                ]
            },
            "context": "用户确认要获取刚才助手整理提及出来的几首曲目",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "下载或索取具体音频文件，直接阻断检索跳入执行层。"
        }
    },
    {
        "input": "这专不错，怎么全集入库",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "当前播放的专辑所有曲目"
                ]
            },
            "context": "想要把当前好评专辑内所有歌曲一键存入个人库中",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "功能性的“入库/收藏专辑”指令意图。"
        }
    },
    {
        "input": "那个背景音乐给我下一下",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "当前提到的背景音乐"
                ]
            },
            "context": "执行下载操作以获得指定的背景BGM",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "用户对于下载和获得实质文件的要求。"
        }
    },
    {
        "input": "提取这首原版伴奏，我存一下",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "当前歌曲的伴奏版本"
                ]
            },
            "context": "保存特殊版本（伴奏）至系统内个人名下",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "获取音频文件相关，意图类别明确。"
        }
    },
    {
        "input": "太困了，我要睡着了，有什么能提神的",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "激情"
            },
            "context": "因为太困想睡觉而反向需求能带来清醒和振奋感的情绪向选曲",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "激情",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "隐含意图极容易被错误分类为“睡觉”场景，但实际上用户诉求是“提神”（对应激情激动情绪）。需要强力借助向量找强节奏音乐。"
        }
    },
    {
        "input": "生活好窒息，想找个没有任何杂音的桃源躲起来",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "治愈"
            },
            "context": "面临巨大的生活压力导致窒息感，极需逃避现实的疗愈氛围和极简音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "治愈",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "抽象情感比喻。图谱限定治愈系词条，其余的“无杂音”、“桃源”交由向量听感兜底提取空灵元素。"
        }
    },
    {
        "input": "今天拿到奖金啦哈哈哈哈哈",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "开心"
            },
            "context": "拿到钱极其狂喜，想听欢快的歌",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "开心",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "边界case，“哈哈哈哈哈”作为隐含情绪标识开朗愉悦状态，而非无意义闲聊。"
        }
    },
    {
        "input": "回忆起过去那些美好的旧时光",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "怀旧"
            },
            "context": "沉湎于往昔岁月的追忆中，需要时代印记强烈的感伤或怀念音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "怀旧",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "纯情感体验陈诉。提取怀旧mood进行全面匹配寻找复古色调单曲。"
        }
    },
    {
        "input": "脑子里乱糟糟的，帮我清空一下",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "平静"
            },
            "context": "遇到思维混乱烦恼，通过平和安宁的音乐实现大脑清空",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "平静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "并非要求聊天，而是寻求缓解状态的手段，此时推荐宁神平静功效音乐最佳。"
        }
    },
    {
        "input": "想要点那种小酒馆里驻唱唱的民谣",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "民谣"
            },
            "context": "推荐类似小酒馆现场环境下的人声伴奏民谣类曲目",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "folk",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "用户指定流派的同时描绘了特定的声学场地环境（小酒馆），除了图谱卡位必须依赖向量来分辨Live版或木吉他编曲味道。"
        }
    },
    {
        "input": "有没有节奏比较重的嘻哈说唱",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "说唱"
            },
            "context": "鼓点和重低音非常突出的说唱Hip-Hop作品推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "hip-hop",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "指定了两个相同实质的流派词并且强调了主观听感“节奏重”，图谱与向量必须联手。"
        }
    },
    {
        "input": "听古典乐，尤其是肖邦那种细腻钢琴曲",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "古典"
            },
            "context": "寻找具有肖邦细腻特质的古典钢琴演奏版本音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": ["肖邦", "Chopin"],
                "graph_genre_filter": "classical",
                "graph_scenario_filter": None,
                "graph_mood_filter": "浪漫",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "提到肖邦多是浪漫主义细腻质感，可作为氛围要求交由向量辨识，并且在分类库里划进 classical。"
        }
    },
    {
        "input": "放点轻快活泼的流行乐",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "流行",
                "mood": "开心"
            },
            "context": "推荐情绪偏向快乐放松同时风格属于大众流行的音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "开心",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "流派（Pop）+ 特定心情氛围（开心/快乐）组合查询。"
        }
    },
    {
        "input": "推荐一首前卫摇滚（Prog Rock），有点迷幻特色的",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "摇滚",
                "mood": "梦幻"
            },
            "context": "寻找含有迷幻元素气质的前卫摇滚长曲",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "rock",
                "graph_scenario_filter": None,
                "graph_mood_filter": "梦幻",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "深度的流派二级分类需求，需要将大类划入rock，“迷幻特色”作为极端声学形容词供向量深度查找兼配“梦幻”情绪。"
        }
    },
    {
        "input": "我想找国内有没有感觉像 Taylor Swift 早期弹吉他的那种女歌手",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "Taylor Swift",
                "region": "内地"
            },
            "context": "寻找国内类似早期泰勒斯威夫特抱着吉他唱非商业民谣/乡村风格的独立女歌手推介",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "Taylor Swift",
                    "泰勒·斯威夫特"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": "Chinese",
                "graph_region_filter": "Mainland China",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "对标典型艺人且加了地区限制寻找“代餐”。依靠图谱做华语限制，依靠向量对比霉霉早期的原声吉他清唱音色。"
        }
    },
    {
        "input": "不要林俊杰了，换个别的男歌手唱同样高难度的",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "林俊杰"
            },
            "context": "推掉林俊杰但延续其难度高亮声线的其他男歌手",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "林俊杰",
                    "JJ Lin"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "含有否定式表达并且带有特定指向艺人寻找相似质感的延展（高难度演唱），利用实体图谱避开林俊杰但用向量寻找对应相似声部。"
        }
    },
    {
        "input": "推点跟薛之谦苦情歌类似的曲子，最好是刚火的新歌",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "薛之谦",
                "mood": "伤感"
            },
            "context": "网络上最近流行的跟薛式伤感情歌路线相似的新作品",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "薛之谦",
                    "Joker Xue"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "伤感",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "类似薛之谦 伤感 流行新歌"
            },
            "reasoning": "具有强时效性（刚火新歌）和歌手对标（薛之谦）的三维复杂搜索，触发联网与向量/图谱协同。"
        }
    },
    {
        "input": "来点和许嵩一样有年代青春感的声音",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "许嵩",
                "mood": "怀旧"
            },
            "context": "寻找具有和许嵩一样千禧年网络青春印记质感的流行歌曲",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "许嵩",
                    "Vae"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "怀旧",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "通过人物实体传达“青春”、“年代感”（映射到怀旧情绪），利用向量搜索同年代音乐风格。"
        }
    },
    {
        "input": "放一首我之前标了喜欢但是好久没听的",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "从用户红心库中挖掘长期未被触碰到的偏好老歌",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "本质还是提取个人相关点赞列表做打底，直接走向图谱User节点数据索取。"
        }
    },
    {
        "input": "根据我的听歌历史给我推几首同类别的",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "根据用户过去的长短期历史沉淀，智能扩展推荐口味相投的歌曲",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "直接引述“我的听歌历史”，系统将获取其近期偏好自动推拉，意图定在favorites下处理。"
        }
    },
    {
        "input": "再来一首昨天我一直在循环的那种风格",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "读取近期的循环播放行为日志作为锚点给新的衍生",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "提及昨天单曲循环等历史用户行为特征指标，由历史链路负责挖掘处理。"
        }
    },
    {
        "input": "在健身房撸铁！需要爆裂的鼓点",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "健身",
                "mood": "热血"
            },
            "context": "应对在健身房举重力量训练等高强度释放过程所需的重击音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "健身",
                "graph_mood_filter": "激情",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "非常典型的具象化特定场景加主观爆裂听觉，图谱负责设定健身分类并辅以向量高能搜索。"
        }
    },
    {
        "input": "正准备入睡，来半小时的纯白噪音或者极简环境音吧",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "睡前",
                "mood": "平静"
            },
            "context": "睡前催眠平复脑波用的无结构化人声和明显旋律的氛围噪音",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "睡前",
                "graph_mood_filter": "平静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "非常需要依靠向量搜索出“极无结构、纯白噪音”的音频资源，配合图谱睡前标签服用极佳。"
        }
    },
    {
        "input": "周末在家大扫除，放点动感流行的华语歌可以跟着唱的那种",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "家务",
                "genre": "流行"
            },
            "context": "做家务搞卫生时用的自带华语易合唱特性的动感流行单曲群",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": "做饭",
                "graph_mood_filter": "开心",
                "graph_language_filter": "Chinese",
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "虽然未明确说做饭，大扫除等居家劳动可平替为打理生活场景，叠加明确的华语、动感和流行等三位一体强标识。"
        }
    },
    {
        "input": "出去散步溜达，听点轻松愉快的",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "散步",
                "mood": "快乐"
            },
            "context": "外出悠闲散步漫游时聆听的无压力轻松曲目",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "散步",
                "graph_mood_filter": "快乐",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "指名道姓给出散步标签，同时附送轻松高兴的情绪基调。"
        }
    },
    {
        "input": "兄弟们开黑打游戏了！战歌起",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "打游戏",
                "mood": "热血"
            },
            "context": "多人游戏聚会开黑时的氛围煽动战歌专属音乐库",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "打游戏",
                "graph_mood_filter": "激昂",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "强游戏场景锚定词打底，同时“战歌”等同于要求极其爆燃、高压热血的听觉体验交给向量处理。"
        }
    },
    {
        "input": "你平常会经常听人吐槽吗",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "用户对系统功能边界和AI自身形象展开探索或者开始求安慰倾诉",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "没有任何直接点歌推歌字眼纯粹社交试探。"
        }
    },
    {
        "input": "不想听歌了，咱俩聊会儿天",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "打断推荐循环，直接要求进入大语言闲谈陪伴状态",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "直抒胸臆中断业务进程发起闲聊邀请意图。"
        }
    },
    {
        "input": "其实我觉得你们 AI 懂个屁的音乐",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "具有攻击性或者讽刺性的闲谈对话内容产生",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "边缘测试或宣泄，不涉及检索指令需经语言管理节点安抚回复。"
        }
    },
    {
        "input": "我该怎么向她表白呢，唉",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "用户遭遇感情瓶颈向助手索取生活建议而不是纯歌单",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "虽然可以发散为浪漫情歌推荐，但这更主要是一句对生活求救的发问，首选闲聊对话安慰再伺机顺口推歌最佳。"
        }
    },
    {
        "input": "那张照片删了三遍还是没舍得，心里空落落的",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "伤感"
            },
            "context": "失恋或失去重要的人导致的心里空虚感",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "伤感",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "强烈的隐喻失恋和思念场景，没有任何歌手歌名，完全交给向量捕捉伤感、落寞的音色氛围配以图谱伤感标签。"
        }
    },
    {
        "input": "连续加了五个通宵的班，感觉身体已经不属于我了",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "疲惫"
            },
            "context": "极度疲劳和过度劳累的身体与心理状态",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "工作",
                "graph_mood_filter": "放松",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "虽然起因是'加班'但核心诉求是缓解极致的'疲惫'，转化图谱情绪为放松，用环境音或轻音乐进行安抚。"
        }
    },
    {
        "input": "明天就要高考了，手心全都是汗",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "焦虑"
            },
            "context": "面对重大事件前夕产生的严重焦虑、紧张情绪",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "平静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "表现出的手心出汗是严重的焦虑症状，音乐推荐应转向反向缓解焦虑的'平静'类型。"
        }
    },
    {
        "input": "刚收到暗恋对象回的信息！！心跳得好快！！",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "兴奋"
            },
            "context": "因感情进展顺利而激发的极致兴奋和甜蜜心情",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "浪漫",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "隐喻明确，属于激动开心的粉红氛围，图谱标签为开心/浪漫，向量找寻轻快活泼的女声或甜蜜流行。"
        }
    },
    {
        "input": "感觉全世界都不懂我，只有关上门的世界是安全的",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "孤独"
            },
            "context": "人际疏离造成的强烈孤独感和安全感缺失",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "孤独",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "深度的防备与社恐孤独心情，提取孤独情绪标签，向深邃的独立音乐或者器乐摇滚方向靠拢。"
        }
    },
    {
        "input": "不知道为什么突然有点想家了，想吃妈妈做的菜",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "思念"
            },
            "context": "异乡人在外由于味觉或场景触发的对家乡和亲人的思念",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "怀旧",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "乡愁和思念倾向于温暖、治愈、怀旧类型的归属感音乐。图谱使用怀旧，向量负责寻找温情的编曲。"
        }
    },
    {
        "input": "不想说话，就想一个人呆着，最好有点声音垫着",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "平静"
            },
            "context": "社交疲劳后寻找可以充当无打扰背景音的静音体验",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "平静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "并非完全的孤独，而是单纯的主动独处和'背景音'需求，平静情绪最能命中。"
        }
    },
    {
        "input": "房租又涨了，工资还是那么点，看着大街上的人觉得自己是个废物",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "忧郁/沮丧"
            },
            "context": "现实生活重压下产生的严重自我怀疑与沮丧情绪",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "忧郁",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "深度的城市丧文化情感诉诸，对应的图谱是忧郁情绪，推荐以悲伤或致郁向民谣/摇滚为主。"
        }
    },
    {
        "input": "老子今天把老板给炒了！太特么爽了！",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "热血/释放"
            },
            "context": "做出了重大反抗决定后的极致痛快与解脱释放感",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "激情",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "反向宣泄代表的是强攻击性或强释放性的激情情绪，需动用向量查找速度极快、失真严重的快感金曲。"
        }
    },
    {
        "input": "看着前任发了结婚的请柬，我连祝福都说不出口",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "悲伤/释怀"
            },
            "context": "得知前度结婚后五味杂陈难以言表的深层悲哀与无力感",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "伤感",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "典型的遗留情感痛点触发，这需要依靠向量找寻富有叙事感和极尽催泪的伤感流行乐。"
        }
    },
    {
        "input": "周末早晨拉开窗帘阳光特别好，想伸个大大的懒腰",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "治愈/惬意"
            },
            "context": "清晨享受阳光时毫无压力的轻松惬意状态",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "治愈",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "纯正面、充满希望的闲适光景。通过图谱获取治愈标签并让向量主攻清脆的原声吉他或大调钢琴。"
        }
    },
    {
        "input": "好烦啊好烦啊这题怎么做都做不对",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "愤怒/急躁"
            },
            "context": "由于解题学习受挫产生的较轻微急躁与暴躁情绪",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "学习",
                "graph_mood_filter": "平静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "虽然明说是烦心，但实际上处于学习打断期更适合用清亮平静的纯音乐或者Lo-fi帮其去火进入专注状态。"
        }
    },
    {
        "input": "翻出高中时候的校服，突然好想以前的同桌",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "怀旧"
            },
            "context": "由老物件触发的对旧日同窗岁月极其单纯的怀念",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "怀旧",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "触发点非常复古明确，即回忆杀。转为图谱情绪中的怀旧要求检索华语青春老歌最佳。"
        }
    },
    {
        "input": "今晚又要一个人面对这四面墙了",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "孤独"
            },
            "context": "对于单身独居在夜晚下产生的寂寥",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "孤独",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "高度具象化地隐秘表达'孤独'，不具备特定流派指向，非常看重声音包围感与向量共鸣。"
        }
    },
    {
        "input": "怎么每天都是一样的事情，活着好像一台机器机器",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "忧郁/麻木"
            },
            "context": "产生严重的异化感和对日复一日生活的厌倦麻木感",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "忧郁",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "厌世感倾向。过滤忧郁情绪寻找那些能穿透僵尸生活状态拥有极强唤醒能力的迷幻或者另类摇滚。"
        }
    },
    {
        "input": "看中好久的口红终于买到啦！",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "开心"
            },
            "context": "小确幸带来的日常轻易满足的兴奋心情",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "开心",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "购物带来的多巴胺分泌属于直接的开心，用图谱锁定流行或者电子的开心跳跃听感。"
        }
    },
    {
        "input": "躺在草地上看云卷云舒，如果时间能停在这一刻就好了",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "放松/治愈"
            },
            "context": "极度亲近大自然时身心完全舒展且无所事事的慵懒时刻",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "田园牧歌般的高质量放松状态，开启图谱放松并让下游寻找充满自然质地的声音比如新纪元或民谣。"
        }
    },
    {
        "input": "气死我了这都是些什么傻逼客户",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "愤怒"
            },
            "context": "职场遭遇奇葩客户气到上头的狂怒状态",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "工作",
                "graph_mood_filter": "愤怒",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "满格的戾气，适合来点核爆级金属或是重型地雷说唱来帮助代偿发泄情绪。"
        }
    },
    {
        "input": "给我放《那些年》，胡夏唱的那首",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "那些年",
                "artist_name": "胡夏"
            },
            "context": "请求播放特定歌手胡夏的主打歌那些年",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "即使包含年代隐喻但实际上这还是一个指名道姓的具体点播任务。"
        }
    },
    {
        "input": "五月天现场万人大合唱的那种 Live 版有没有",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "五月天",
                "mood": "热血"
            },
            "context": "希望寻找五月天具备极强现场合唱感和互动热情的Live音源",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "五月天",
                    "Mayday"
                ],
                "graph_genre_filter": "rock",
                "graph_scenario_filter": None,
                "graph_mood_filter": "激情",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "指定了实体，但非常强调场馆物理回声（Live）和氛围（大合唱），这简直就是为向量定制的声学兜底搜索。"
        }
    },
    {
        "input": "我想下暴雨在高速上开夜车的时候听那种史诗感的",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "开车",
                "mood": "激情"
            },
            "context": "下暴雨开夜车的极端压抑却又宏大高危的驾车氛围需史诗音乐搭配",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "开车",
                "graph_mood_filter": "激情",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "场景非常极致（飙车/开夜车），加上“史诗感”，翻译到图谱为开车+激情，由于是极难命中的听觉要求必须依赖向量捕捉大编制管弦乐特质。"
        }
    },
    {
        "input": "就想听一些类似 The Weeknd 充满迷幻跟合成器复古感的",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "The Weeknd"
            },
            "context": "找到能在合成器流行乐语境下散发糜烂复古气质对标威肯的作品",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "The Weeknd",
                    "盆栽哥"
                ],
                "graph_genre_filter": "r&b",
                "graph_scenario_filter": None,
                "graph_mood_filter": "梦幻",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "有明确的参照艺人和具体的编曲风格倾向（迷幻跟合成器复古），图谱限制双语，向量承担极具体的硬件声学特征索取。"
        }
    },
    {
        "input": "刚把工作辞了准备漫无目的地去大西北跑一圈",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "旅行",
                "mood": "治愈/释放"
            },
            "context": "辞职后的长途西北公路旅行追求广袤释放的听音体验",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "folk",
                "graph_scenario_filter": "旅行",
                "graph_mood_filter": "治愈",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "隐喻了非常清晰的“远方旅行”（活动），并透露出离职解脱之绪，因此旅行叠加治愈是最佳图谱分配方案。"
        }
    },
    {
        "input": "最近这一个月有没有新的能打的K-pop团体冒出来",
        "output": {
            "intent_type": "search",
            "parameters": {
                "genre": "流行",
                "region": "韩国"
            },
            "context": "查询近一个月韩国爱豆流行乐坛的新晋实力团体与神曲",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": "Korean",
                "graph_region_filter": "Korea",
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "韩国 K-pop 最近一个月 新女团男团 出道神曲"
            },
            "reasoning": "包含强烈的“最近这一个月”、“新团体”等无法在静态库里保证不滞后的资讯探求！立刻激活Web联查机制！"
        }
    },
    {
        "input": "你谈过恋爱没有？怎么给我放这么苦的歌",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "对机器人产生反讽和吐槽交互且对曲目负反馈",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "不仅在直接质疑AI的恋爱经验（纯闲聊），同时委婉提出了前面推的歌风格不讨喜，应转Chat由大语言安抚调侃。"
        }
    },
    {
        "input": "感觉自己就像一个不停旋转但是找不到方向的陀螺",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "迷茫"
            },
            "context": "由于生活失去目标和方向而产生的强烈迷茫与无力感",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "安静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "深度的隐喻表达。将“迷茫”转换为安静或治愈的图谱标签，通过向量提供沉浸式思考的声学体验。"
        }
    },
    {
        "input": "为什么付出那么多最后还是只换来一句对不起",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "伤感/委屈"
            },
            "context": "感情挫败导致的心碎、不甘心与委屈悲痛",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "伤感",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "强烈的被抛弃的失恋语境，没有给出具体歌名，依靠向量寻找催泪苦情歌进行极度悲伤和伤感的情绪代偿。"
        }
    },
    {
        "input": "今天看电影《霸王别姬》哭成了狗",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "悲伤/震撼"
            },
            "context": "看虐心电影之后情绪久久不能平复的沉醉与悲伤",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "悲伤",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "由于外在艺术作品触发的伤痛。虽然有实体《霸王别姬》，但本质是为了延续那种悲剧艺术感，因此锁定悲伤情绪并由向量提供宏大影视配乐。"
        }
    },
    {
        "input": "连续下了一周的雨，满脑子都是霉味",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "烦闷/压抑"
            },
            "context": "因为长时间阴雨天气引发的生理性厌倦和心理极度压抑",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "治愈",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "边界情绪，用户处于烦闷期，需要以对立情绪“治愈/轻快”或者干净空灵的声学向量（如清脆吉他）来打破这种压抑。"
        }
    },
    {
        "input": "这破班是一天也上不下去了！想砸电脑！",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "暴躁/愤怒"
            },
            "context": "被工作压力和职场环境逼到爆发边缘的狂暴状态",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "工作",
                "graph_mood_filter": "愤怒",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "狂躁情绪非常极化。利用图谱的愤怒标签筛选出极端重金属、说唱等发泄型大声压级爆裂曲目。"
        }
    },
    {
        "input": "刚看了下银行卡余额，连下个月泡面都吃不起了",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "无奈/凄凉"
            },
            "context": "因极度贫穷和经济困窘导致的现实无力感与辛酸",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "忧郁",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "虽然说的是吃泡面没钱，但这反映了纯纯的忧郁和心酸心情。用忧郁打底寻找底层叙事的都市民谣或蓝调。"
        }
    },
    {
        "input": "买到了明天回家的车票！可以见到爸妈啦！",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "激动/开心"
            },
            "context": "离家游子即将返乡团聚的极度喜悦与迫切激动",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "旅行",
                "graph_mood_filter": "开心",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "快乐和激动的巅峰。归类为开心/兴奋情绪，并由向量配以极高正向能量的律动性音乐。"
        }
    },
    {
        "input": "一个人在医院挂水看着药液一滴一滴掉，周围安静得可怕",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "孤独/脆弱"
            },
            "context": "生病独自就医时由于寂静环境渲染出的极其脆弱的孤独封闭感",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "治愈",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "边界场景：“安静得可怕”代表内心的不安感，需要的是反向的“治愈/温暖”陪伴情绪来缓解生病时的弱势心理。"
        }
    },
    {
        "input": "被自己蠢哭了，怎么会犯这么低级的错误哎",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "懊恼/自责"
            },
            "context": "因个人失误陷入较轻度但密集的懊恼自我怀疑情绪中",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "平静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "懊恼时不需要过激的悲伤，而更需要平复心情。故映射到图谱中平静或治愈，向量寻找和缓的节拍。"
        }
    },
    {
        "input": "刚刚站在体重秤上，看到那个数字感觉人生都失去了意义",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "破防/沮丧"
            },
            "context": "减肥失败或容貌身材焦虑带来的崩溃与破防",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "幽默又辛酸的日常沮丧。不用提供极端悲歌，反而应该用轻松治愈（放松）的曲调进行调侃式的身心安抚。"
        }
    },
    {
        "input": "晚上喝了点小酒，微醺状态看窗外的霓虹灯，感觉整座城市都在漂浮",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "迷幻/沉醉"
            },
            "context": "在酒精作用下产生的特有城市夜归人浪漫感与迷幻沉沦感",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "梦幻",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "高度诗意化的情绪表述！利用图谱“梦幻”锁区，再仰仗向量大力出奇迹，抓取City Pop或是慵懒爵士这种迷醉听感。"
        }
    },
    {
        "input": "收到大学录取通知书的时候，我爸居然背着我偷偷抹眼泪了",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "感动/温馨"
            },
            "context": "被极小且珍贵的亲情片段击中，极度感动与温情涌上心头",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "治愈",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "不具备特定名词实体的长段落隐喻场景。感动等同于重大的正面治愈情绪（治愈/温暖）。"
        }
    },
    {
        "input": "刚刚看完了那本虐恋小说的大结局，简直喘不过气",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "揪心/痛心"
            },
            "context": "因为代入文艺作品导致的深度感同身受的压抑心痛",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "看书",
                "graph_mood_filter": "伤感",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "提到了小说大结局（看书活动）与极度难受（伤感情绪），需要交响配乐感强且极度催泪的向量声学支撑。"
        }
    },
    {
        "input": "外面狂风大作好像世界末日要来了，但我窝在被窝里感觉好安全",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "安逸/幸福"
            },
            "context": "外部恶劣环境与内部安适环境形成的强烈对比带来的窝心安全感",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "睡前",
                "graph_mood_filter": "放松",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "非常复杂的隐蔽情感对比。“世界末日”是假象核心是“好安全”的安逸放松，可以带睡前标签找极简Lofi白噪音。"
        }
    },
    {
        "input": "我不要听那些咿咿呀呀的流行情歌了，给我上点能在音乐节上甩断脖子的",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "摇滚室外/金属"
            },
            "context": "否定了抒情曲风，追求高能、极具现场破坏力的极端音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "metal",
                "graph_scenario_filter": None,
                "graph_mood_filter": "激情",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "边界case，“甩断脖子”是重金属圈黑话，指向极端金属流派及现场Live质感，向量与图谱共同兜转到高压音轨。"
        }
    },
    {
        "input": "好久没关注欧美圈了，现在 Billborad 榜单上第一的是谁",
        "output": {
            "intent_type": "search",
            "parameters": {
                "region": "欧美"
            },
            "context": "查询当期欧美流行音乐公信榜的最新排名冠军",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "Billboard Hot 100 最新榜单 第一名 冠军单曲"
            },
            "reasoning": "典型的极强时效性榜单查询（Billboard当前第一），必须依赖联网外抓不能靠本地检索！"
        }
    },
    {
        "input": "我觉得你比我前男友还懂我",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "由于推荐结果极大满足预期引发的用户对AI助手的超高评价和类人情感投射",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "并不为了让AI去放情歌或者前男友喜欢的歌，而是纯粹的人机交互好评肯定谈话环节。"
        }
    },
    {
        "input": "这首歌副歌那个吉他Solo太无敌了，有类似这种段落的吗",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "摇滚"
            },
            "context": "对正在播放的曲目特定乐器solo演奏技巧表达极度肯定并索取同类",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "rock",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "索求的是纯乐器声音特质体验（吉他Solo）。没有提到任何歌手，归类为流派推荐（摇滚），完全依靠向量在摇滚大类底层深挖纯听感相似曲。"
        }
    },
    {
        "input": "赶紧把这首放进我的自建列表【发呆专用】里",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "当前歌曲"
                ]
            },
            "context": "将好评曲目显式收藏打标到用户的特定歌单中保存",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "收藏入库等偏向资产处置范畴的用户指令，不需要搜索推荐介入。"
        }
    },
    {
        "input": "老王乐队的那首《我还年轻我还年轻》",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "我还年轻我还年轻",
                "artist_name": "老王乐队"
            },
            "context": "确切歌手及确切曲名的点播播放要求",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "不管歌名叫什么，只要明确带书名号及歌手，都是Play触发点。"
        }
    },
    {
        "input": "一个人在晚上开车上高架桥，风呼呼吹，来点带感的",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "开车",
                "mood": "激情"
            },
            "context": "夜间高速驾驶场景下追求速度感和带感节奏的听觉刺激",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "开车",
                "graph_mood_filter": "激情",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "强烈的公路驾驶画面感极易映射为图谱的'开车'场景，'带感'指代节奏强烈的'激情'情绪，向量发挥寻找电子或摇滚车载音响的作用。"
        }
    },
    {
        "input": "早高峰挤在沙丁鱼罐头一样的三号线里，救命，给我点喘息的声音",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "通勤",
                "mood": "平静"
            },
            "context": "极度拥挤且令人窒息的早高峰地铁通勤场景",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "通勤",
                "graph_mood_filter": "平静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "幽默又写实的沙丁鱼地铁通勤隐喻，由于环境极其嘈杂烦躁，需要利用向下的向量提取如清脆钢琴曲或白噪音等带来'平静'呼吸感的音乐。"
        }
    },
    {
        "input": "在炖一锅要熬三个小时的牛腩汤，找点慢节奏的调调陪我",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "做饭",
                "mood": "放松"
            },
            "context": "长时间居家慢炖烹饪过程中的悠闲与放松状态",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "做饭",
                "graph_mood_filter": "放松",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "充满着烟火气的做饭画面（炖牛腩汤）。不适合激烈的歌曲，需要设定放松的图谱情绪和慢节奏轻声学的向量组合。"
        }
    },
    {
        "input": "晚上点好蜡烛开好红酒了，等下女朋友就来，赶紧弄点气氛组的",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "约会",
                "mood": "浪漫"
            },
            "context": "精心布置的私密居家浪漫烛光晚餐约会准备环节",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "约会",
                "graph_mood_filter": "浪漫",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "蜡烛、红酒等要素强烈指涉约会活动，气氛组要求极高。图谱绑定浪漫和约会，交由向量去匹配性感的 R&B 或慢摇爵士。"
        }
    },
    {
        "input": "在阳台的摇椅上翻那本厚厚的百年孤独，有没有合适的背景音",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "看书"
            },
            "context": "由于深阅读《百年孤独》发起的适合陪伴看书专注的背景配乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "看书",
                "graph_mood_filter": "平静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "非常典雅的读书场景，不需要有人声打扰。启用图谱'看书'过滤器，用向量找到厚重但不过于突兀的大提琴或无声环境音。"
        }
    },
    {
        "input": "戴上耳机准备开始冲艾尔登法环的最终Boss了！",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "打游戏",
                "mood": "热血"
            },
            "context": "进入高难度动作游戏终极挑战前的战前准备与亢奋状态",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "打游戏",
                "graph_mood_filter": "激昂",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "画面直击魂系游戏玩家冲刺阶段。对应打游戏标签，音乐急需提供史诗感和极度燃炸（激昂）的情绪供血。"
        }
    },
    {
        "input": "现在在海边踩着沙子吹海风，不知道听什么好",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "旅行",
                "mood": "治愈"
            },
            "context": "在自然环境如海边漫步时的放松度假心境",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "散步",
                "graph_mood_filter": "治愈",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "踩沙子、吹海风其实是一种大自然氛围里的散步或旅行，适合非常空灵治愈的氛围系小调。推荐图谱'散步' + '治愈'。"
        }
    },
    {
        "input": "刚把瑜伽垫铺好拉伸，需要一点禅意",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "健身",
                "mood": "平静"
            },
            "context": "准备进行冥想或室内柔韧性瑜伽训练",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "健身",
                "graph_mood_filter": "平静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "不同于撸铁，铺瑜伽垫虽同属'健身'大类，但情绪要求截然相反（平静/禅意），需严重依赖向量去分辨出颂钵等静心声学。"
        }
    },
    {
        "input": "今晚又是肝 PPT 到凌晨两点的硬仗，来点防打瞌睡的",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "工作",
                "mood": "激情"
            },
            "context": "面临长久深夜加班文案工作的抗疲劳作战指令",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "工作",
                "graph_mood_filter": "激情",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "肝PPT对应工作打码，但需求不是白噪音，而是明确的'防打瞌睡'（激情、振奋），让向量去锁定快板电子乐等。"
        }
    },
    {
        "input": "正在洗澡，花洒水声有点大，放点动静不一般的让我在浴室能跟着扭",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "洗澡",
                "mood": "开心"
            },
            "context": "私密的浴室洗澡时间内渴望极大声音动态和欢乐节奏的个人KTV",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "洗澡",
                "graph_mood_filter": "开心",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "生活感极强的非常规场景洗澡，由于水声掩蔽加上想“扭”，必须用图谱抓取开心标签并通过向量匹配鼓点重、穿透力强的欢乐舞曲。"
        }
    },
    {
        "input": "把猫抱在怀里撸，看窗外下大雪",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "睡前/居家",
                "mood": "治愈"
            },
            "context": "极度舒适、温暖的冬季室内容易催生睡意的居家治愈画面",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "睡前",
                "graph_mood_filter": "治愈",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "没有任何直接活动明示，“撸猫看雪”指向极静的安逸。可归类为睡前或打理生活的治愈系小调。"
        }
    },
    {
        "input": "骑着共享单车在林荫道下穿梭，这时候应该听什么",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "骑行",
                "mood": "放松"
            },
            "context": "充满青春气息或生活悠闲感的户外自行车骑行游荡",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "散步",
                "graph_mood_filter": "放松",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "由于骑行多类似散步，追求微风拂面的清爽感。图谱挂靠散步/旅行大类，由向量查找充满阳光质感的清新流行曲。"
        }
    },
    {
        "input": "宿舍熄灯了可是我完全睡不着，舍友的呼噜声震天响",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "睡前",
                "mood": "平静"
            },
            "context": "夜间遭遇强烈噪音干扰导致的失眠状态",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "睡前",
                "graph_mood_filter": "平静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "虽然有抱怨呼噜声的成分，但核心是在宿舍上床（睡前活动）后企图利用大自然白噪音或者环境音（向量）隔绝对抗呼噜并助眠。"
        }
    },
    {
        "input": "刚刚在酒吧外面吹冷风等代驾，给来点配电影画面的",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "路途/等待",
                "mood": "忧郁"
            },
            "context": "深夜醉酒或微醺后在街头吹风等待时的电影感寂寞抽离体验",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "孤独",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "边界场景。等代驾的闲散时间，结合吹冷风具有典型的都会寂静感（忧郁/孤独），需要极具氛围感和BGM属性的音乐向量。"
        }
    },
    {
        "input": "跟朋友在露营地生起了篝火，这时候适合大家一起听的",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "聚会",
                "mood": "温馨"
            },
            "context": "户外露营圈友聚会的松弛、快乐与轻度狂欢",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "folk",
                "graph_scenario_filter": "聚会",
                "graph_mood_filter": "开心",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "露营和篝火是非常标准的户外派对（聚会活动），需要图谱卡位加上偏向Acoustic吉他弹唱的民谣（向量加权）以契合氛围。"
        }
    },
    {
        "input": "一个人在超市推着车挑速冻水饺，挑着挑着就哭了",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "买菜/生活",
                "mood": "伤感"
            },
            "context": "极具生活实感和烟火气中反衬出的大城市孤单致郁破防瞬间",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "居家",
                "graph_mood_filter": "孤独",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "典型的微小活动场景（挑选速冻水饺）附带极端情感（哭了，孤独）。图谱映射到偏生活的场景并叠加极致的孤独标签寻找催泪向单曲。"
        }
    },
    {
        "input": "拿上画板坐在公园湖边准备写生，有什么能激发灵感",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "艺术创作",
                "mood": "平静"
            },
            "context": "在自然环境中进行美术绘画等静心创作时的配底需求",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "学习",
                "graph_mood_filter": "平静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "写生活动可映射为专注度极高的学习或阅读类目。不需要人声歌词污染听觉，通过向量寻求古典或纯音乐背景。"
        }
    },
    {
        "input": "家里办大扫除擦玻璃呢，来点跟劳动号子一样特别上头的",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "家务",
                "mood": "激昂"
            },
            "context": "强体力居家家务活中需要增强洗剪吹、清扫节奏干劲的听音辅助",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "做饭",
                "graph_mood_filter": "激昂",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "虽然说是劳动号子，但大扫除本质对应“做饭/居家”场景。极其“上头”的心情要求动感的快歌（开心/激昂）甚至拉动土嗨流行。"
        }
    },
    {
        "input": "这盘棋下到残局了，老头子我有点卡壳，放点能帮着思索的曲艺吧",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "下棋/休闲",
                "genre": "曲艺"
            },
            "context": "带有浓郁国风老年人群体在消磨时间下象棋卡壳时需要传统的配乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "folk",
                "graph_scenario_filter": None,
                "graph_mood_filter": "平静",
                "graph_language_filter": "Chinese",
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "罕见活动（下棋）伴随对国潮戏曲的渴求。“曲艺”可作为流派近似词处理（民乐/folk/戏腔），图谱加语言过滤以及向量识别特色乐器。"
        }
    },
    {
        "input": "这首歌是谁翻唱的，比原唱好像还牛逼一点",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "对于刚听完的重置版本音乐发表极高认可度和惊讶疑问",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "用户并不是要重新查询其他歌曲，而是就在针对当下这一曲目的人声及版权发问。适合语言大模型自己接盘解释这是哪个翻唱。"
        }
    },
    {
        "input": "刚刚错过了那段前奏，给我倒回去并且把这首下到手机里",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "当前播放曲目"
                ]
            },
            "context": "极细致的控制播放进度结合明确的文件占有下载企图",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "控制播放端并不是推荐任务，“把这首下到手机里”构成了纯正的acquire意图。"
        }
    },
    {
        "input": "帮我看看现在网易云热歌榜里有没有好听的古风音乐",
        "output": {
            "intent_type": "search",
            "parameters": {
                "genre": "古风"
            },
            "context": "通过指定具体平台近期的热歌指标，跨库寻找含有中国传统声学元素的流派歌曲",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "folk",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": "Chinese",
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "网易云音乐 最新 热歌榜 古风"
            },
            "reasoning": "复合查询！既指定了平台榜单的热度时效性（必须Web），又框定了中国风（古风，图谱选folk+中文+向量判别）。"
        }
    },
    {
        "input": "切一首孙燕姿的我怀念的",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "我怀念的",
                "artist_name": "孙燕姿"
            },
            "context": "毫无悬念的组合拳点歌切歌过程",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "常规且绝对固定的显式歌曲调用指令。"
        }
    },
    {
        "input": "放一首我一个月以前听过，当时单曲循环了好久的那个摇滚",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "带有极强时间线索和历史行为筛选特征的个人库召回请求",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "虽然提到了具体的时间（一个月前）操作（循环）和类别（摇滚），但因为主体核心指向“我听过”的历史，故判为红心历史动作交由历史工具检索。"
        }
    },
    {
        "input": "有没有霉霉的那种乡村时期的老歌",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "Taylor Swift",
                "genre": "乡村"
            },
            "context": "寻找泰勒·斯威夫特早期创作的乡村音乐类型歌曲",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "Taylor Swift",
                    "泰勒·斯威夫特"
                ],
                "graph_genre_filter": "country",
                "graph_scenario_filter": None,
                "graph_mood_filter": "怀旧",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "用户使用了中文外号“霉霉”，图谱需要映射到标准的双语实体 [Taylor Swift, 泰勒·斯威夫特]；同时带有'乡村'流派以及'老歌'（怀旧）的查询条件。"
        }
    },
    {
        "input": "搜一下酷玩乐队有什么经典的英伦摇滚",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "Coldplay",
                "genre": "摇滚",
                "region": "欧美"
            },
            "context": "查询酷玩乐队推出的经典英伦摇滚流派作品",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "Coldplay",
                    "酷玩乐队"
                ],
                "graph_genre_filter": "rock",
                "graph_scenario_filter": None,
                "graph_mood_filter": "怀旧",
                "graph_language_filter": "English",
                "graph_region_filter": "Western",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "指定了中文乐队名（需带出英文原名实体 Coldplay），流派'摇滚'，语言地区附带英伦（Western/English），'经典'可用怀旧或向量兜底。"
        }
    },
    {
        "input": "那谁，A妹最近这两年有什么新专或者单曲不",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "Ariana Grande"
            },
            "context": "查询爱莉安娜·格兰德近两年来发行的新专辑和单曲",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "Ariana Grande",
                    "爱莉安娜·格兰德"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "Ariana Grande A妹 最近两年 新专辑 新单曲"
            },
            "reasoning": "使用了中文粉圈外号“A妹”，必须识别出 Ariana Grande 实体；“最近这两年”极度依赖时效性，必须触发 web_search 检索。"
        }
    },
    {
        "input": "花花华晨宇有哪些特别高音或者嘶吼的现场版",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "华晨宇",
                "mood": "激昂"
            },
            "context": "搜索包含华晨宇(花花)展现极高音域或者爆发力嘶吼的演出现场音频",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "华晨宇",
                    "Hua Chenyu"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "激情",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "花花+华晨宇组合识别，难点在于'高音'、'嘶吼'和'现场版'，没有任何图谱能直接框死，极其依赖下游向量声学的匹配与情绪检索(激情)。"
        }
    },
    {
        "input": "帮我找盆栽哥(The Weeknd)最好听的一首 R&B",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "The Weeknd",
                "genre": "R&B"
            },
            "context": "在歌手盆栽哥的作品里挑选流派为节奏布鲁斯的最优曲目",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "The Weeknd",
                    "盆栽哥"
                ],
                "graph_genre_filter": "r&b",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "双语实体夹杂查询（盆栽哥/The Weeknd），直接明确了流派R&B。图谱双过滤操作。"
        }
    },
    {
        "input": "查一下魔力红 Maroon 5 早期的流行电子乐",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "Maroon 5",
                "genre": "电子",
                "mood": "怀旧"
            },
            "context": "探索魔力红乐队早年带有明显电子流行风格的作品",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "Maroon 5",
                    "魔力红"
                ],
                "graph_genre_filter": "electronic",
                "graph_scenario_filter": None,
                "graph_mood_filter": "怀旧",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "中英混杂双语实体。对作品有三个限定：'早期'带来怀旧时间滤镜，'流行电子'明确为electronic大类，启用向量保证辨识度。"
        }
    },
    {
        "input": "我想找那个谁，结石姐 Jessie J 在歌手上唱的那首歌",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "Jessie J"
            },
            "context": "回忆并搜索Jessie J在综艺节目《歌手》中演唱的著名曲目",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "Jessie J",
                    "结石姐"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "Jessie J 结石姐 歌手 综艺节目 演唱曲目"
            },
            "reasoning": "这是一个具有极强跨界背景知识的搜索：涉及到特定综艺栏目事件，本地库可能未打综艺标签，建议开启web_search结合图谱兜底查找。"
        }
    },
    {
        "input": "听厌了周杰伦，有没有稍微带点他那个周氏 R&B 味道的别的湾湾男歌手",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "周杰伦",
                "genre": "R&B",
                "region": "台湾地区"
            },
            "context": "拒绝周杰伦本人的作品，但渴望寻找同时代带有相似华语R&B质感的台湾地区其他男歌手",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "周杰伦",
                    "Jay Chou"
                ],
                "graph_genre_filter": "r&b",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": "Chinese",
                "graph_region_filter": "Taiwan",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "极度复杂的否定加延展找平替！图谱需要识别原歌手作为反向剔除锚点，同时加入地域（Taiwan）、流派（R&B）、语言（Chinese）限制，最终依靠向量捕获周氏R&B旋律感。"
        }
    },
    {
        "input": "想找像 Billie Eilish 碧梨那种嗓音沙哑但是又很拽的女声",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "Billie Eilish",
                "mood": "放松"
            },
            "context": "由于喜欢碧梨慵懒、沙哑且极具个性的女声，进而索要相近调性的其他音乐人创作",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "Billie Eilish",
                    "碧梨"
                ],
                "graph_genre_filter": "indie",
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "双语实体。'沙哑'、'拽'是一种特殊的声学音色和听歌情绪（放松/慵懒），这属于强向量辅助任务，通常属于独立(indie)流行类别。"
        }
    },
    {
        "input": "感觉国内有没有像西城男孩 Westlife 这样和声很好听的男团",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "Westlife",
                "region": "内地"
            },
            "context": "借助经典欧美男团西城男孩的优质和声特征，映射寻觅国内的类似男子演唱组合的歌曲",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "Westlife",
                    "西城男孩"
                ],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": "Chinese",
                "graph_region_filter": "Mainland China",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "用海外实体做模板，限制地域为国内（Mainland China）、语言默认中文，指望依靠向量在流行库里大浪淘沙找多轨和声音源。"
        }
    },
    {
        "input": "防弹少年团 BTS 现在休团了，推点跟他们舞曲风格类似的爆裂男团KPop",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "BTS",
                "genre": "流行",
                "region": "韩国"
            },
            "context": "基于防弹少年团的高强度舞曲风格，寻找其他韩国男团的平替",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "BTS",
                    "防弹少年团"
                ],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "激情",
                "graph_language_filter": "Korean",
                "graph_region_filter": "Korea",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "中英实体'防弹少年团/BTS'。不仅框死KPop（韩国+流行+韩语语种），还提出了'舞曲'和'爆裂'等极其带劲的声学需求，因此必须通过图谱加向量配合推荐发散。"
        }
    },
    {
        "input": "不要林夕写的词，给我放首跟李宗盛一样能把词讲进男人心里的歌",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "李宗盛",
                "mood": "伤感"
            },
            "context": "反向排除林夕作词的情节，转而渴求李宗盛般直击中年男性内心、叙事感极强的极品老歌代餐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "李宗盛",
                    "Jonathan Lee",
                    "林夕"
                ],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "伤感",
                "graph_language_filter": "Chinese",
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "极其刁钻的高级意图。不仅涉及多个实体（李宗盛作为正面推荐锚点，林夕作为反向剔除锚点），还涉及到文本深度（讲进男人心里），可映射为有阅历的“伤感/叙事”，利用向量做内容感知。"
        }
    },
    {
        "input": "有没有类似陈绮贞 Cheer Chen 那种抱把木吉他随随便便也能唱得很空灵的",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "陈绮贞",
                "mood": "平静"
            },
            "context": "通过陈绮贞代表的极简木吉他弹唱流派，寻找同样拥有空灵声线嗓音特色的治愈系独立曲目",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "陈绮贞",
                    "Cheer Chen"
                ],
                "graph_genre_filter": "folk",
                "graph_scenario_filter": None,
                "graph_mood_filter": "平静",
                "graph_language_filter": "Chinese",
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "中英名字，提到木吉他直接翻译为民谣（folk）类型配上安静空灵的女声特征（由平静情绪和向量把关发掘）。"
        }
    },
    {
        "input": "周深最近给哪部国产电视剧或者动画电影唱了 OST 吗？能不能播一下",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "周深",
                "region": "内地"
            },
            "context": "查询歌手近期的最新跨界演唱动向（主要是影视剧原声OST）并尝试播放",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "周深",
                    "Zhou Shen"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": "Chinese",
                "graph_region_filter": "Mainland China",
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "周深 最近 电视剧 动画电影 OST 主题曲 2025"
            },
            "reasoning": "对动态变化极快的周边发行业务(近期新OST)发起搜寻，本地图谱百分百过时，立刻启用 web_search 进行实体和外网信息的穿透联动。"
        }
    },
    {
        "input": "我要听皇后乐队 Queen 那首在波西米亚狂想曲电影里火得一塌糊涂的六分钟的长歌",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "artist_name": "Queen",
                "song_name": "Bohemian Rhapsody"
            },
            "context": "带有极强暗示以及乐队信息的准确代表作点播",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "虽然没带书名号也描述得很长，但这其实指名道姓点出了是波西米亚狂想曲和Queen乐队的长作，本质属于长尾点歌 play_specific_song_online。"
        }
    },
    {
        "input": "老是被林肯公园或者绿日那种欧美朋克洗脑",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "朋克",
                "region": "欧美"
            },
            "context": "由几个知名乐队牵引出的要求大规模播放欧美朋克流派歌曲的长列表需求",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "Linkin Park",
                    "林肯公园",
                    "Green Day",
                    "绿日乐队"
                ],
                "graph_genre_filter": "punk",
                "graph_scenario_filter": None,
                "graph_mood_filter": "激情",
                "graph_language_filter": "English",
                "graph_region_filter": "Western",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "用户列举了多个实体作引子，由于主体是想听“欧美朋克”，归因给流派搜索更为准确。图谱开启多端锁定。"
        }
    },
    {
        "input": "不要欧美也不要日韩，就想要那种土生土长带点我们民族乐器的老派摇滚",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "摇滚",
                "region": "内地",
                "mood": "怀旧"
            },
            "context": "提出强否定的跨地区限制后，谋求带有中国民族元素编曲的本土特色老牌摇滚乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "rock",
                "graph_scenario_filter": None,
                "graph_mood_filter": "怀旧",
                "graph_language_filter": "Chinese",
                "graph_region_filter": "Mainland China",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "经典复合流派+地域过滤！排斥外语环境（限制死 Mainland China 和 Chinese），老派译为怀旧，而“带点我们民族乐器”是无法靠单纯标签做出来的，极大依赖向量下海捞针。"
        }
    },
    {
        "input": "你觉得 AI 写出来的歌曲能够打败莫扎特吗？",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "对音乐行业未来和技术哲学发展的闲聊式探讨",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "即便是提到了实体莫扎特，也不是要听古典乐，而是抛出了极高的形而上聊天请求。"
        }
    },
    {
        "input": "有没有那种节奏感特别强的韩流舞曲",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "流行舞曲",
                "region": "韩国",
                "mood": "开心"
            },
            "context": "寻找具有强烈舞蹈节奏的韩国流行（K-Pop）音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "激情",
                "graph_language_filter": "Korean",
                "graph_region_filter": "Korea",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "“韩流舞曲”可以直接解析为：地区（Korea）、语种（Korean）、大流派（Pop）以及极具动感（激情），多重图谱过滤加向量强节奏搜寻。"
        }
    },
    {
        "input": "想听听英伦摇滚，不要太吵，有点忧郁气质的那种",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "摇滚",
                "region": "欧美",
                "mood": "忧郁"
            },
            "context": "寻找具有独特阴郁和旋律化气质的轻度英伦摇滚乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "rock",
                "graph_scenario_filter": None,
                "graph_mood_filter": "忧郁",
                "graph_language_filter": "English",
                "graph_region_filter": "Western",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "流派筛选为'摇滚'，极度明确了地区'欧美'，而且反向限定了不要太吵（避免重金属），加入情绪限定词'忧郁'来指导向量剔除过噪的音轨。"
        }
    },
    {
        "input": "给点国内的校园民谣吧，感觉回到了大学草坪上一样",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "民谣",
                "region": "内地",
                "mood": "怀旧"
            },
            "context": "怀旧向的中国大陆早期校园纯真时期民谣吉他弹唱",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "folk",
                "graph_scenario_filter": None,
                "graph_mood_filter": "怀旧",
                "graph_language_filter": "Chinese",
                "graph_region_filter": "Mainland China",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "复合了流派（folk）、地区（Mainland China）以及情感隐喻（校园草坪=青春怀旧）的三维交叉推荐。"
        }
    },
    {
        "input": "最近很迷那种赛博朋克风的合成器波 Synthwave 电子乐",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "电子",
                "mood": "梦幻"
            },
            "context": "寻找具有复古未来主义以及强烈霓虹色彩听感的Synthwave电子派系",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "electronic",
                "graph_scenario_filter": None,
                "graph_mood_filter": "梦幻",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "非常专业的流派黑话'Synthwave'。大类进入electronic，依靠向量的强悍声学理解能力找到迷幻/梦幻的复古合成器音轨。"
        }
    },
    {
        "input": "来点慵懒的爵士乐，适合下雨天喝咖啡听的",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "爵士",
                "mood": "治愈/放松"
            },
            "context": "雨天特供的具有室内氛围感和极度松弛感的传统或现代慵懒爵士乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "jazz",
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "完美组合了流派（Jazz）+ 情绪（慵懒/放松）+ 场景（听雨喝咖啡，近似于平静治愈），交由双索共同出击。"
        }
    },
    {
        "input": "有没有气势极其磅礴悲壮的交响乐，史诗级的那种",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "古典",
                "mood": "激昂/悲伤"
            },
            "context": "大型管弦乐团演奏的极具压迫感和史诗悲凉感的古典结构交响乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "classical",
                "graph_scenario_filter": None,
                "graph_mood_filter": "激昂",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "虽然没直说古典，但这属于管弦交响(Classical)；由于同时带了雄壮激昂（气势磅礴）和凄凉感（悲壮），这是非常考验向量理解能力的极端曲风。"
        }
    },
    {
        "input": "推点早年间台湾特产的那些咬字不清的 R&B 快歌可以吗",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "R&B",
                "region": "台湾地区",
                "mood": "开心"
            },
            "context": "回顾千禧年间华语乐坛黄金时期由周氏唱腔带火的台湾节奏布鲁斯快节奏曲目",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "r&b",
                "graph_scenario_filter": None,
                "graph_mood_filter": "怀旧",
                "graph_language_filter": "Chinese",
                "graph_region_filter": "Taiwan",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "流派(R&B) + 地域(台湾) + 时代感(早年间=怀旧) + 曲风(快歌=动感)。极其丰满的限制网。"
        }
    },
    {
        "input": "我要那种歌词极其狠毒、beat巨大声的欧美硬核说唱",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "说唱",
                "region": "欧美",
                "mood": "愤怒"
            },
            "context": "寻找具有强攻击性歌词以及厚重鼓点伴奏的西方匪帮或硬核Hip-Hop",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "hip-hop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "愤怒",
                "graph_language_filter": "English",
                "graph_region_filter": "Western",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "流派(Hip-hop) + 地区(欧美/Western) + 情绪表现(歌词狠毒代表着攻击性与愤怒)。"
        }
    },
    {
        "input": "有没有充满牙买加风情的雷鬼音乐",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "雷鬼",
                "mood": "放松"
            },
            "context": "寻找具有强烈加勒比海岛听感和反拍吉他扫弦特征的Bob Marley风格的雷鬼乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "reggae",
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "小众但极具辨识度的流派（Reggae），通常带有热带岛屿和放松自由的听觉特征。"
        }
    },
    {
        "input": "推荐能在舞会上让人忍不住扭胯的拉丁热情舞曲",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "拉丁",
                "mood": "开心/激情"
            },
            "context": "具有极为狂热和性感的南美风情、适合社交舞蹈的快节奏拉丁流行乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "latin",
                "graph_scenario_filter": "聚会",
                "graph_mood_filter": "激情",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "流派（拉丁）结合使用场景（舞会/聚会）以及主体体验（热情/兴奋/能扭胯），多管齐下。"
        }
    },
    {
        "input": "一个人喝闷酒，来首绝望到底的密西西比三角洲蓝调",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "蓝调",
                "mood": "孤独/忧郁"
            },
            "context": "追求极其纯正、充满泥土气息且极度悲凉苦痛的早期根源Blues",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "blues",
                "graph_scenario_filter": None,
                "graph_mood_filter": "孤独",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "专有名词（密西西比蓝调）对应流派Blues，加上了极致致郁的情境刻画（喝闷酒、绝望到底对应孤独/忧郁）。"
        }
    },
    {
        "input": "推点美国西雅图或者德克萨斯那种带着班卓琴的老乡村味道",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "乡村",
                "region": "欧美"
            },
            "context": "具备传统配器（班卓琴）与北美浓烈西部开阔感的老牌乡村音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "country",
                "graph_scenario_filter": None,
                "graph_mood_filter": "怀旧",
                "graph_language_filter": "English",
                "graph_region_filter": "Western",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "点出了乐器（班卓琴）以及地域特色。翻译到约束框图为 Western/English + Country，以及老味道（怀旧）。"
        }
    },
    {
        "input": "我现在需要极高 BPM 的硬核 Techno 电子，没有的话 Trance 也可以",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "电子",
                "mood": "激昂"
            },
            "context": "寻找电子舞曲分支中最高能的硬核techno或致幻trance",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "electronic",
                "graph_scenario_filter": None,
                "graph_mood_filter": "激情",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "对于极其刁钻的子流派诉求，全部圈定于electronic大类！同时提供高能（激情）向量配合，绝不是单纯字符串匹配。"
        }
    },
    {
        "input": "国内的各种古风歌曲听腻了，有没有那种日本的物哀美学的和风纯音乐",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "民谣/国风",
                "region": "日本",
                "mood": "平静"
            },
            "context": "舍弃国产古风体系，探索日式传统乐器营造的悲寂物哀和风气质（纯器乐）",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "folk",
                "graph_scenario_filter": None,
                "graph_mood_filter": "平静",
                "graph_language_filter": "Japanese",
                "graph_region_filter": "Japan",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "极其精彩的否定与重新定向组合！图谱必须框住日本/日语以及和风（归为泛folk底色），物哀美学则由向量结合平静标签抓取！"
        }
    },
    {
        "input": "不想要带人声的，推点能让我安静下来的后摇（Post-Rock）",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "摇滚",
                "mood": "平静"
            },
            "context": "具有庞大声景但没有歌手演唱的安静氛围系长篇后摇滚作品",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "rock",
                "graph_scenario_filter": None,
                "graph_mood_filter": "平静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "否定人声。这要求使用摇滚(rock)的大类进行限制，但由于情绪标签是“平静”，将很大程度遏制传统摇滚的出现，向量专门捕获纯乐器吉他噪音墙。"
        }
    },
    {
        "input": "我觉得你这推荐的简直就是垃圾",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "严重的负向对抗情绪输入，没有带检索信息",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "直接触发纯闲谈的AI大头回应，不能乱调用流派和点播。"
        }
    },
    {
        "input": "好不容易周末休个假，给我随便来点好听的，不用挑了",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "放松"
            },
            "context": "因为终于迎来的周末而发起的极度随意的休闲放松无脑泛推请求",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "虽然说是“随便”，但依然隐含有“周末休假”带来的解压与“好听”的主观诉求，归为放松情绪让系统随机找点好听的发车。"
        }
    },
    {
        "input": "最近在抖音上那个'我们一起学猫叫'的那首歌叫啥来着，查查看",
        "output": {
            "intent_type": "search",
            "parameters": {
                "song_name": "学猫叫"
            },
            "context": "由于记忆模糊只依靠残存的几句短视频现象级歌词发起的寻歌查询",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "抖音 我们一起学猫叫 歌名"
            },
            "reasoning": "带有片段歌词的不确定式寻常搜寻。因并不百分百确定本体，直接依赖外部互联网歌词大本营（web_search）进行容错查找！！"
        }
    },
    {
        "input": "有没有那种九十年代的香港老电影里经常放的、带点萨克斯的粤语情歌",
        "output": {
            "intent_type": "search",
            "parameters": {
                "region": "香港地区",
                "language": "粤语",
                "genre": "流行",
                "mood": "怀旧"
            },
            "context": "寻找具有浓缩九十年代港风电影质感的经典粤语流行老歌（特定乐器：萨克斯）",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "怀旧",
                "graph_language_filter": "Cantonese",
                "graph_region_filter": "Hong Kong",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "非常典型的四维复合意图：地区(Hong Kong) + 语言(Cantonese) + 流派(pop) + 年代情绪(怀旧)。萨克斯与老电影质感则留给向量兜底。"
        }
    },
    {
        "input": "现在在居酒屋喝酒，搞点日式的轻音乐或者三味线弹奏垫一垫",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "聚会/喝酒",
                "region": "日本",
                "genre": "国风/传统"
            },
            "context": "日料店或居酒屋背景下需要营造日式传统氛围的纯器乐BGM",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "folk",
                "graph_scenario_filter": "聚会",
                "graph_mood_filter": "放松",
                "graph_language_filter": "Japanese",
                "graph_region_filter": "Japan",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "三维叠加：活动(喝酒=聚会/放松) + 地域(Japan/Japanese) + 乐器流派(轻音乐/三味线打入folk)。"
        }
    },
    {
        "input": "推点像五月天早年那种极其热血、能让人在操场上狂奔的英伦摇滚偏流行的",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "五月天",
                "genre": "吉他流行/摇滚",
                "mood": "激昂"
            },
            "context": "基于五月天早期作品特征寻找高动能的华语或英伦吉他流行摇滚",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "五月天",
                    "Mayday"
                ],
                "graph_genre_filter": "rock",
                "graph_scenario_filter": None,
                "graph_mood_filter": "激昂",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "艺术家实体作引子(Mayday) + 情绪(早年/热血=激昂/怀旧) + 流派(摇滚)。"
        }
    },
    {
        "input": "有没有节奏很快很洗脑的英文电音舞曲",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "language": "英语",
                "genre": "电子",
                "mood": "开心/激情"
            },
            "context": "寻找具有强烈律动感、快速BPM的欧美系英文电子舞曲",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "electronic",
                "graph_scenario_filter": None,
                "graph_mood_filter": "激情",
                "graph_language_filter": "English",
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "经典三复合：语种语调(English) + 明确大类(electronic) + 节奏指标(快/洗脑=激情)。"
        }
    },
    {
        "input": "不想听国产民谣那种无病呻吟的，给我整点欧美大胡子老头唱的老派乡村",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "乡村",
                "region": "欧美",
                "mood": "怀旧"
            },
            "context": "拒绝国产城市民谣，转向极其纯正硬核的北美怀旧乡村音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "country",
                "graph_scenario_filter": None,
                "graph_mood_filter": "怀旧",
                "graph_language_filter": "English",
                "graph_region_filter": "Western",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "否定(排斥Mainland/folk) -> 肯定(欧美/country/怀旧)。非常精彩的情感情景反转。"
        }
    },
    {
        "input": "健身房蹬动感单车呢，来点能卡上BPM120的心跳声的重低音Hip-hop",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "健身",
                "mood": "激昂",
                "genre": "说唱"
            },
            "context": "在高强度有氧运动中渴求节拍固定(BPM120)、低重音极强的嘻哈说唱",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "hip-hop",
                "graph_scenario_filter": "健身",
                "graph_mood_filter": "激昂",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "极其精准的三维限制：场景过滤(健身) + 流派过滤(hip-hop) + 情绪表现(激昂/动感)，并通过向量去捕获所谓的'重低音BPM120'。"
        }
    },
    {
        "input": "准备睡觉了，放点小语种的催眠轻哼，千万别有鼓点",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "睡前",
                "language": "小语种",
                "mood": "平静"
            },
            "context": "睡前免打扰模式下，渴望听不懂歌词词义（避免大脑处理语言）且绝无节奏打击乐的轻声哼唱",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "folk",
                "graph_scenario_filter": "睡前",
                "graph_mood_filter": "平静",
                "graph_language_filter": "Others",
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "极其刁钻的要求。防语言干扰（配置Others并过滤中英），场景（睡前），情绪（平静），排除了鼓点（纯靠向量把关）。"
        }
    },
    {
        "input": "开车自驾游去新疆，窗外全是大戈壁，放点西北风情的大气民谣或者世界音乐",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "旅行/开车",
                "region": "内地/西北",
                "genre": "国风/传统"
            },
            "context": "大西北公路旅行场景下对富有地域辽阔感中国传统或民谣音乐的点播",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "folk",
                "graph_scenario_filter": "开车",
                "graph_mood_filter": "激昂",
                "graph_language_filter": "Chinese",
                "graph_region_filter": "Mainland China",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "非常棒的交织：自驾/大戈壁=开车/旅行，西北风情=Mainland_China+folk，大气=激昂或空旷辽阔的向量感知。"
        }
    },
    {
        "input": "帮我找一下断眉(Charlie Puth)跟那个谁，赛琳娜合唱的那个当年霸榜的单曲",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "Charlie Puth, Selena",
                "mood": "怀旧"
            },
            "context": "通过提供两个国外歌手的中英文互译实体词，交叉检索他们曾经合作过的冠军热单（We Don't Talk Anymore）",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "Charlie Puth",
                    "断眉",
                    "Selena",
                    "赛琳娜"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": "English",
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "Charlie Puth 断眉 Selena 赛琳娜 合唱 霸榜 单曲 We Don't Talk Anymore"
            },
            "reasoning": "双实体提取极其复杂！包含两个人物（断眉+Charlie Puth / 赛琳娜）以及事件描述（合唱且当年霸榜），本地库不一定有合唱挂链，必须上web_search确保万无一失。"
        }
    },
    {
        "input": "网抑云时间到了，给我整点极其致郁的俄罗斯后朋克",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "摇滚/后朋",
                "region": "俄罗斯",
                "mood": "忧郁"
            },
            "context": "黑夜引发的网络深夜降级孤独症患者对东欧苦寒地带极度冰冷、绝望的后朋音乐的诉求",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "rock",
                "graph_scenario_filter": None,
                "graph_mood_filter": "忧郁",
                "graph_language_filter": "Others",
                "graph_region_filter": "Western",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "高强度网络黑话综合。“网抑云”=深夜致郁(忧郁/孤独+睡前)。区域限制(俄罗斯/Others)，流派(后朋/分类为rock底层细分)。"
        }
    },
    {
        "input": "刚生完一场大气，想听点充满攻击性但是又不是摇滚乐的那种交响乐，就是纯砸场子那种",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "愤怒/激昂",
                "genre": "古典"
            },
            "context": "在极度愤怒需发泄情形下，排斥常规金属重型乐，转求带有极强听觉攻击力的古典管弦交响狂想曲",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "classical",
                "graph_scenario_filter": None,
                "graph_mood_filter": "愤怒",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "反向排斥（不要摇滚），锁定流派（交响乐=classical）。情绪不仅是宏大，带有了极端的攻击发泄性（愤怒/激昂）。"
        }
    },
    {
        "input": "最近很喜欢听那种女声很清透的日系爵士，Bossa Nova那种调调的",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "爵士",
                "region": "日本",
                "language": "日语"
            },
            "context": "指定小众但高度具象化的日流Bossa Nova/City Pop系女声爵士单曲",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "jazz",
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": "Japanese",
                "graph_region_filter": "Japan",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "明确框定语种和地区（日语/日本），大类选定jazz（巴萨诺瓦Bossa Nova也是其轻音乐分支），女声清透由向量负责捕获。绝妙的三维组合标签。"
        }
    },
    {
        "input": "一个人在巴黎街头喝着左岸咖啡却刚失恋，放点最能让人碎掉的香颂吧",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "旅行/独处",
                "mood": "伤感",
                "region": "欧洲/法国"
            },
            "context": "在异国他乡（法国）遭遇感情重大创伤极度孤独下对本地极强时代感音乐(香颂)的渴求",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "folk",
                "graph_scenario_filter": "散步",
                "graph_mood_filter": "忧郁",
                "graph_language_filter": "Others",
                "graph_region_filter": "Western",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "多重元素极度堆叠。失恋(忧郁/伤感) + 巴黎街头(Others语言/Western地域) + 香颂(类似民谣或老派流行folk/pop)。"
        }
    },
    {
        "input": "你之前放的那首粤语歌里面混了一段英语Rap的，那首到底叫什么名字，我还想听一遍",
        "output": {
            "intent_type": "search",
            "parameters": {
                "language": "粤语/英语",
                "genre": "说唱"
            },
            "context": "针对曾经记忆中一首极其复杂特征（粤英切换跨界Rap）的旧歌发起的再次重检索追溯",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "hip-hop",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": "Cantonese",
                "graph_region_filter": "Hong Kong",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "具有强烈的历史聊天特征，但由于带出了特征(粤英双语、Rap=hiphop)，更倾向于让向量根据这极其刁钻的复合语种和流派在库里寻回。"
        }
    },
    {
        "input": "不想要欧美说唱那种全都是车子票子脏话的，有没有国内注重歌词诗意和押韵的那种Old School",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "说唱",
                "region": "内地",
                "mood": "平静/深沉"
            },
            "context": "否定了商业化Trap的铜臭味和攻击性，转向追求歌词有深度具有人文关怀的华语老学校说唱",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "hip-hop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "平静",
                "graph_language_filter": "Chinese",
                "graph_region_filter": "Mainland China",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "强烈的否定拉踩：不要欧美(Western)不要炫富(激情/愤怒)，只要中文Old School（Mainland China + hip-hop + 走心/平静）。多重复合对抗！"
        }
    },
    {
        "input": "来点",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "放松"
            },
            "context": "由于用户并未提供任何实质性线索，触发兜底式的随性放松推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "边界情况。用户输入极短且毫无意义(属于泛泛的“来点”)。不应当报错，而是引导到最安全的'放松'情绪随机播放。"
        }
    },
    {
        "input": "放首歌",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "开心/治愈"
            },
            "context": "最底层的兜底笼统指令，无约束",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "治愈",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "“放首歌”类似口语的随机来首日推，默认采用正向情绪（治愈/开心）作为推荐发端。"
        }
    },
    {
        "input": "啊啊啊啊啊啊啊啊啊啊啊",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "激动/狂躁"
            },
            "context": "因情绪极限失控而产生的纯符号/拟声词重复输入",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "激昂",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "虽然是无意义重复，但连续感叹号或'啊'象征着极端的情绪发泄，可映射为激昂/狂躁的向量。"
        }
    },
    {
        "input": "wdnmd",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "愤怒"
            },
            "context": "国内极具代表性的网络粗口缩写，表达极度的不快与愤怒",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "愤怒",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "非常本地化的骂人黑话边界，由于体现了高度愤怒特征，故将其导向愤怒/重型宣泄音乐。"
        }
    },
    {
        "input": "那个，就是那个，那个调调是怎么唱的来着",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "用户处于极度语言组织不能的状态，缺乏任何有效搜索和推荐实体",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "废话边界。没有任何实体，连只言片语的歌词都没有，必须交由大模型主轮(general_chat)去反问用户提示细节。"
        }
    },
    {
        "input": "帮我搜一下州杰纶的一路箱南",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "周杰伦",
                "song_name": "一路向南"
            },
            "context": "具有多处极其严重的同音错别字和故意恶搞的寻歌",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "周杰伦",
                    "Jay Chou"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "周杰伦 一路向南"
            },
            "reasoning": "考验模型自我纠错。把州杰纶(周杰伦)、一路箱南(一路向南)纠正过来，然后启动web_search搜索歌单或利用图谱直出。"
        }
    },
    {
        "input": "林卷结的浆南",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "artist_name": "林俊杰",
                "song_name": "江南"
            },
            "context": "高密度离谱错字（将林俊杰写成林卷结，将江南写成浆南）",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "边界鲁棒性测试，直接点歌（play意图），并将错字纠正为正确词元(林俊杰-江南)。"
        }
    },
    {
        "input": "难听死了，我不听，全部全删掉！",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "由于算法推送不合格导致的极大抱怨与情绪排斥",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "强抱怨。不属于任何点歌及流派诉求，交由闲聊大模型用情感安抚接盘。"
        }
    },
    {
        "input": "不要这种带歌词的，不要太吵的，也不要那个什么交响乐，就普普通通的行不行",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "平静"
            },
            "context": "连续列出多条否决条件后，给出的诉求极其泛化('普普通通的')",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "folk",
                "graph_scenario_filter": None,
                "graph_mood_filter": "平静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "排查了(Vocals)、(摇滚重低音)、(Classical)，最后落在平静轻音乐或极其质朴的指弹吉他上，靠图谱folk和向量清脆感。"
        }
    },
    {
        "input": "你能不能给我唱个《葫芦娃》听听",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "试图让文字AI助理去用语音拟声唱歌（虽然底层是推荐系统）",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "超纲需求边界。由于系统仅仅能召回和播放歌曲，并没有虚拟数字人自我演唱的能力，所以触发闲谈婉拒或者调侃。"
        }
    },
    {
        "input": "帮我算一下 124 乘以 56 等于多少",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "完全脱离了音乐业务范畴的数学计算闲聊指令",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "无关业务打发到 general_chat 处理即可。"
        }
    },
    {
        "input": "弄点俺们东北那种二人转的底子那啥的老歌",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "曲艺/民乐",
                "region": "内地",
                "mood": "开心/怀旧"
            },
            "context": "极具地域方言口语化特色的本土戏曲老歌搜索",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "folk",
                "graph_scenario_filter": None,
                "graph_mood_filter": "怀旧",
                "graph_language_filter": "Chinese",
                "graph_region_filter": "Mainland China",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "把东北二人转归为广义曲艺（folk/Traditional），地区绑定Mainland China，要求老歌(怀旧)，以及向量去抓东北唢呐大鼓的高亮特征。"
        }
    },
    {
        "input": "有没有那种，就是，dong dong dong，然后突然 ziga ziga 的那种",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "电子",
                "mood": "激情"
            },
            "context": "极其滑稽的手工口技拟声词寻找特定鼓点变奏的曲目",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "electronic",
                "graph_scenario_filter": None,
                "graph_mood_filter": "激情",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'dong dong'代表重型底鼓，'ziga'代表电音里的搓碟或者合成器失真，完美重叠于Electronic(电子)极其动感(激情)的流派。"
        }
    },
    {
        "input": "放一首最吵的最安静的最极端的舒缓流行金属",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "语无伦次。将完全相悖的形容词糅合并抛出不可能完成的任务",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "故意调流用的错乱测试（又吵又安静，极端又舒缓）。交由 general_chat 去吐槽用户自相矛盾。"
        }
    },
    {
        "input": "...",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "孤独/沉默"
            },
            "context": "以沉默符号暗示无言的寂寥或是无语",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "孤独",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "将'...'等效于孤独或沉默的情绪。为其播放极其安静的环境音或孤寂的民谣。"
        }
    },
    {
        "input": "现在 Apple Music 国区排第一的是哪首歌",
        "output": {
            "intent_type": "search",
            "parameters": {
                "region": "内地"
            },
            "context": "查询极其具体的某流媒体平台(Apple Music)当下的地区实时榜首曲目",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": "Chinese",
                "graph_region_filter": "Mainland China",
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "Apple Music 中国区 国区 最新 热歌榜 第一名 冠军单曲"
            },
            "reasoning": "强依赖时外网信息的查询。因为图谱每天都在变动，不应当死抠本地库，直接派往 web_search 大海捞针获取今日最热资讯。"
        }
    },
    {
        "input": "昨天格莱美拿到年度最佳专辑的是谁，放一下那张专的主打歌",
        "output": {
            "intent_type": "search",
            "parameters": {},
            "context": "由于现实世界刚刚发生的全球顶尖音乐奖项颁布（格莱美），导致的用户搜索及点播请求",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "昨天 最新一届 格莱美 Grammy 年度最佳专辑 获奖者 主打歌"
            },
            "reasoning": "具备绝大信息差的新闻追溯(昨天、格莱美、年度专辑)，本地完全无法命中，靠 web 搜出实体后再去播放流媒体链接。"
        }
    },
    {
        "input": "周杰伦刚才在 Ins 上发的那首新歌试听叫什么",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "周杰伦"
            },
            "context": "追踪头部歌手在海外社交媒体上极短时间内的宣发动作（新歌首曝）",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "周杰伦",
                    "Jay Chou"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "周杰伦 Jay Chou Instagram ins 刚才 最新发布的 新歌 试听片段"
            },
            "reasoning": "高强度时效！虽然锚定了实体（周杰伦），但是由于他“刚才发在Ins上”，这属于外网实时动态，强制触发 web_search 匹配爬虫数据。"
        }
    },
    {
        "input": "有没有 Blackpink 她们这个月刚刚打歌的那首",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "Blackpink"
            },
            "context": "韩流团体在回归期(这个月)参与打歌节目的主打首秀单曲搜索",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "Blackpink",
                    "粉墨"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "Blackpink 粉墨 这个月 最新打歌 回归 主打歌 单曲"
            },
            "reasoning": "“打歌”和“这个月”双极化时效需求。必须识别实体女团名，辅以强大的 web 爬虫寻获。"
        }
    },
    {
        "input": "抖音上那个跳科目三用的贼火的电音是啥",
        "output": {
            "intent_type": "search",
            "parameters": {
                "genre": "电子"
            },
            "context": "借助短视频平台极具热度但不规范的网络爆梗（科目三）反查背景音乐",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": "electronic",
                "graph_scenario_filter": None,
                "graph_mood_filter": "激情",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "抖音 科目三 舞蹈 贼火的 背景音乐 BGM 电音"
            },
            "reasoning": "梗文化的极致。“科目三”本来是考驾照，但在抖音转为土嗨舞蹈。本地缺乏梗百科，必须启用 web_search 进行中文网语义降维匹配电音实体。"
        }
    },
    {
        "input": "《繁花》电视剧里面唐嫣出场放的那首上海话的老歌",
        "output": {
            "intent_type": "search",
            "parameters": {
                "language": "上海话",
                "mood": "怀旧"
            },
            "context": "基于近期爆款影视剧《繁花》的剧中特定人物(唐嫣)登场时的方言经典OST",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [
                    "唐嫣"
                ],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "怀旧",
                "graph_language_filter": "Chinese",
                "graph_region_filter": "Mainland China",
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "繁花 电视剧 唐嫣 汪小姐 出场 背景音 上海话 老歌 插曲"
            },
            "reasoning": "非常棒的长难句结合影视剧热点(繁花)！没有任何歌名及原唱歌手(只知道是唐嫣主演的角色)。图谱拦截不住这种网剧爆梗，直接推给 Web Search 解决。"
        }
    },
    {
        "input": "刚过去的巴黎奥运会开幕式，有一段重金属乐队跟断头王后合唱的那个配乐",
        "output": {
            "intent_type": "search",
            "parameters": {
                "genre": "金属/摇滚"
            },
            "context": "极具画面震撼感和话题度的近期国际大型体育盛事（巴黎奥运）现象级融合音乐现场找回",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": "metal",
                "graph_scenario_filter": None,
                "graph_mood_filter": "激昂",
                "graph_language_filter": None,
                "graph_region_filter": "Western",
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "巴黎奥运会 开幕式 重金属乐队 Gojira 断头王后 玛丽安托瓦内特 合唱 配乐"
            },
            "reasoning": "这是世界级的社会事件检索，而非纯朴的听歌曲库。依靠强大的Web大模型拆解出法国乐队Gojira，而非在音乐图谱里迷失。"
        }
    },
    {
        "input": "B站上那个二创鬼畜视频《大梦一场的董明珠》原曲是什么",
        "output": {
            "intent_type": "search",
            "parameters": {},
            "context": "试图获取被视频网站弹幕或二次创作修改后的爆款鬼畜视频真正的原版原唱音频",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [
                    "董明珠"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "B站 哔哩哔哩 鬼畜视频 二创 大梦一场的董明珠 原曲 原唱"
            },
            "reasoning": "二次元恶搞文化检索。“董明珠”并非歌手，必须利用 Web 确认《大梦》原曲实为《大漠》（或者民谣等情况），只有外网词源才能搞定二次元鬼畜。"
        }
    },
    {
        "input": "林俊杰上周六在台北小巨蛋演唱会上唱了一首没发过的 Demo",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "林俊杰",
                "region": "台湾地区"
            },
            "context": "探寻上周六发生的演唱会活动中首秀且未数字发行过的歌手小样记录",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "林俊杰",
                    "JJ Lin"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": "Taiwan",
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "林俊杰 JJ Lin 上周六 台北 小巨蛋 演唱会 未发行 Demo 新歌"
            },
            "reasoning": "因为未发行！本地流媒体不可能入库，所以必定属于网搜新闻咨询件，去抓别人饭拍录音资源。"
        }
    },
    {
        "input": "最近特别火的那个网飞剧《降世神通》实写版，里面片尾曲那种战歌",
        "output": {
            "intent_type": "search",
            "parameters": {
                "genre": "原声带/古典",
                "mood": "激昂"
            },
            "context": "利用新出的网剧（近期很火）作为引子想要获取具有'战歌'(激昂、史诗感)的片尾曲",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": "classical",
                "graph_scenario_filter": None,
                "graph_mood_filter": "激昂",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "网飞 Netflix 降世神通 真人版 实写版 片尾曲 战歌 原声 OST"
            },
            "reasoning": "混合场景。影视剧片尾曲寻找必定用 web_search，加上'战歌'(对应向量的激昂史诗古典派)，做到了外网定点与本地库声学推荐的结合可能。"
        }
    },
    {
        "input": "我记得2008年北京奥运会有一首几百人合唱的，不是北京欢迎你，是哪首",
        "output": {
            "intent_type": "search",
            "parameters": {},
            "context": "基于具有长远历史特征(2008)的重大活动发起检索，并具有强条件排他性要求",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "激昂",
                "graph_language_filter": "Chinese",
                "graph_region_filter": "Mainland China",
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "2008年 北京奥运会 群星合唱 几百人合唱 -北京欢迎你"
            },
            "reasoning": "这是一道推理排除题。本地图谱不会记录'多少人合唱了哪首奥运曲'等长文咨询，外派Web，且明确把'北京欢迎你'当成不包含关键词处理。"
        }
    },
    {
        "input": "今年春晚上面那个魔术表演的背景音乐，听着有点像印度歌",
        "output": {
            "intent_type": "search",
            "parameters": {
                "region": "印度"
            },
            "context": "对新年春晚特备节目上极其短暂出现的冷门伴奏（被感受为印度风）发起猎奇溯源",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "今年 春晚 魔术表演 刘谦 背景音乐 BGM 印度歌"
            },
            "reasoning": "典型的实时热点(春晚魔术)，背景音乐(BGM)完全没名字，强制 Web Search 根据网民自发扒带分享进行倒推检索。"
        }
    },
    {
        "input": "放一下我昨晚睡觉前单曲循环了一针晚的那首纯音乐",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "用户请求播放其个人历史播放记录中具有极高近期权重（昨晚循环整晚）的特定曲目",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "强烈的个人历史记录指代（昨晚、单曲循环、一整晚）。这不需要公域大模型做图谱和向量推荐，而是必须调用推荐历史个人化工具条线。"
        }
    },
    {
        "input": "把我收藏夹最底下，就是好几年前刚注册那时候标红心的老歌随便随机几首",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "基于用户账户注册早期建立的云端喜爱收藏列表进行随机考古播放",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "明确指向'我的收藏夹'和'标了红心'的内部库。不需要做任何流派向量检索，只管调用个人红心资产接口即可。"
        }
    },
    {
        "input": "去年网易云音乐年度报告里，我听得最多的那首",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "根据外部平台的个人年度听歌报告数据记录来点播年度Top 1单曲",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "年度报告Top1本质上属于个人听歌历史榜首，因为是'我听得最多'，强关联为个人足迹推荐模型处理。"
        }
    },
    {
        "input": "不管什么流派，反正只要是我平时喜欢听的，随便来点",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "极度依赖算法对用户长线偏好画像的无脑个人电台推荐",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "排斥了流派，'平时喜欢听的'极其精准地落在了历史爱好的泛推上，这就是典型的进入纯个人电台工具链的绝杀句子。"
        }
    },
    {
        "input": "刚刚这首前奏简直神了！赶紧给我下载到本地无损格式",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "当前播放曲目"
                ]
            },
            "context": "因为某段旋律(前奏神了)触发的极强无损音频文件占有欲和本地化下载操作",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'下载到本地'是标准的执行资产转移管理(acquire)动作，无需推荐。"
        }
    },
    {
        "input": "把薛之谦的《动物世界》和《认真的雪》打包加到我那个叫‘KTV必点’的歌单里面去",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "薛之谦 动物世界",
                    "薛之谦 认真的雪"
                ]
            },
            "context": "批量化地将多首明确已知的曲目指定放入用户自行创建的特定标题歌单资产中",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "非常明确的资产操作指引，将薛之谦的两首歌装入个人歌单。song_queries正确获取复数歌曲。"
        }
    },
    {
        "input": "这专是在QQ音乐还是哪里独家？我要花钱买这首数字单曲 支持一下",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "当前播放曲目"
                ]
            },
            "context": "对正在播放的有版权数字单曲表达强烈的购买意愿以支持偶像或内容创作者",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "花钱购买/支持打榜等所有权资产发生货币或积分交换的行为都属于超级 acquire_music 意图。"
        }
    },
    {
        "input": "快快快，把这首歌立刻放进我的‘绝对不删’红心列表",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "当前播放曲目"
                ]
            },
            "context": "紧急将目前听到的一首极为切中口味的随机推荐曲目标记为常驻红心最高权限名单",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'放进列表'即资产标记操作，属于 Acquire 范畴下的资产整理动作。"
        }
    },
    {
        "input": "播一首李荣浩的《李白》",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "李白",
                "artist_name": "李荣浩"
            },
            "context": "最标准化无歧义且带有歌手信息和书名号确切名称的点唱",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "不仅有歌手，且带有明确歌名《李白》，点唱指令极其坚固。"
        }
    },
    {
        "input": "我要听迈克尔杰克逊的 Billie Jean，立刻马上",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "Billie Jean",
                "artist_name": "Michael Jackson"
            },
            "context": "急迫地要求播放欧美流行之王迈克尔的传奇英文名曲",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "即使没有书名号，但由于“歌手的+具体的英文专有名词歌名”，仍然构成了极强烈的专属精准点歌条件(play_specific_song)。"
        }
    },
    {
        "input": "可以给我放一首《遇见》吗，我是说孙燕姿原版那个，不要翻唱",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "遇见",
                "artist_name": "孙燕姿"
            },
            "context": "要求精确播放极其著名的一首老歌且严格指派必须是原唱歌手版本",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "由于原唱和歌名全部框死(孙燕姿/遇见)，甚至声明拒绝翻唱盲盒(否定了search查找泛结果)，这是必须保证精准投喂的原点播放问题。"
        }
    },
    {
        "input": "麻烦切歌到Eason陈奕迅的那首 富士山下",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "富士山下",
                "artist_name": "陈奕迅"
            },
            "context": "在播放队列中主动切流，直接切换至陈奕迅的指定粤语金曲",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "明确“歌手+歌曲”的点名制操作，属于切歌中的play强指令。"
        }
    },
    {
        "input": "听说周杰伦出了一首新歌叫《红颜如霜》，有这个歌吗？",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "红颜如霜",
                "artist_name": "周杰伦"
            },
            "context": "用户对已知且新近发布，甚至带着书名号的具体曲目发起试探性查询听歌指令",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "边界纠结情况。虽然是'有这歌吗'貌似search，但其实这已经连歌名（红颜如霜）带人（周杰伦）全部说出来了，因此本质诉求就是想点播这首歌，而不是想看列表。"
        }
    },
    {
        "input": "帮我看看那个什么，汪苏泷的那个叫啥...小星星？有的话给我放一遍",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "小星星",
                "artist_name": "汪苏泷"
            },
            "context": "犹豫、结巴后回忆起准确歌名及原唱的即刻点唱要求",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "虽然说话吞吞吐吐，但是实质上的信息(汪苏泷/小星星)依然组成了完整的点唱大招矩阵。这不属于推荐，不需要走图谱。"
        }
    },
    {
        "input": "你知道披头士乐队一共拿过多少次格莱美吗",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "针对音乐界鼻祖级乐队的生平荣誉及客观事实发起的百科式提问",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "纯正的音乐百科提问。用户意在获取知识而不在于听披头士的歌，大模型自带的知识库足以回答，判定为 general_chat。"
        }
    },
    {
        "input": "说唱音乐里面的 Flow 具体指的是什么意思啊，我老是听人说",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "对于特定音乐流派(嘻哈圈)黑话专业术语(Flow)的概念解析求助",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "名词解释需求。无关任何搜歌逻辑，引导 LLM 回答 Flow 代表说唱中的节奏停顿与韵律编排即可。"
        }
    },
    {
        "input": "为什么周杰伦那么喜欢用 4536251 这个和弦走向",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "基于知名歌手作品中高频出现的乐理编曲框架(和弦级数)发起的深度剖析",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "强乐理知识解答。虽然带了周杰伦这个实体，但这显然是让 AI 去论述这套和弦在流行乐中的优势和受众爽点，而非点歌。"
        }
    },
    {
        "input": "古典乐里面的奏鸣曲式一般分哪几个部分",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "关于西方传统古典乐理结构基础构成部分的学术探讨",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "音乐基础理论教育科普范畴，解答呈示部、展开部与再现部。"
        }
    },
    {
        "input": "你觉得陶喆和王力宏巅峰时期谁的 R&B 编曲更高级一点",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "对同时代两位华语R&B天王的音乐制作水准提出对比评判的主观性刁难",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "拉踩性质的评测问题，两个实体出现仅是为了业务对比，系统应端水或者客观拆解各自的编曲特色。"
        }
    },
    {
        "input": "这首歌听起来跟 Ed Sheeran 的 Shape of You 旋律好像啊，算抄袭吗",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "对正在播放歌曲的旋律相似度提出侵权性质疑",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "用户在反馈听感并提出法理性疑问，要求模型客观阐明音乐相似度和抄袭小节法则，安抚或解答用户的听觉错觉。"
        }
    },
    {
        "input": "你这推荐算法是不是换人写了，今天推的每一首都绝杀！",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "推荐系统极致命中用户偏好引发的极高评价与激动赞许的反馈",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "夸奖！主语言模型只需沾沾自喜并感谢用户的赏识，并询问是否需要继续。"
        }
    },
    {
        "input": "早啊，每天早上听你这电台已经变成我的习惯了，今天也要加油哦",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "带有人机日久生情羁绊感的高亲和度晨间每日问候互动",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "寒暄打气，无需硬推歌，模型只需回应温柔暖心的早晨祝福即可。"
        }
    },
    {
        "input": "如果你是一个真人 DJ，你最想在哪里办自己的音乐节",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "把AI拟人化发起的浪漫畅想式开放型问答探讨",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "纯角色扮演。这既不是意图搜歌也不是业务执行，是绝佳的模型自由发挥土壤。"
        }
    },
    {
        "input": "刚才放的那首歌封面好吓人，大晚上的差点没把我送走",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "对专辑视觉封面风格过度惊悚压抑提出抱怨后怕",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "用户并不是要停止播放或者请求切歌（暂时），而是主观的情感宣泄吐槽。让大模型进行陪笑安抚。"
        }
    },
    {
        "input": "网易云和QQ音乐哪个的版权比较多点",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "针对音乐流媒体行业竞品软件的版权壁垒差异发问",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "这是行业客观事实分析。不在业务操作内，纯闲谈科普。"
        }
    },
    {
        "input": "我下个月要去冰岛看极光了，但是我不知道要准备什么厚度的衣服",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "将音乐助手作为旅游攻略万能百科使用而发起的北欧极寒出行装备提问",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "极其脱轨。系统可以基于大模型常识简单告诉他需要防风冲锋衣，并顺水推舟问一句“要不要我在你收拾行李时放点北欧后摇？”（将脱轨转化回主业务！"
        }
    },
    {
        "input": "今天晚上吃红烧肉还是炖排骨，你帮我选一下",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "晚饭选择困难症依赖AI抛硬币的废话文学",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "让大模型随意决定一个即可完成安抚。"
        }
    },
    {
        "input": "那张专辑最后卖了多少钱",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "具有极为强烈的上文依赖的、掐头去尾的询问特定商品销量的追问",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "没有任何可用的推荐特征，这必须交给 LLM 根据上文对话管理去调取历史，在当前无任何参数的 Planner 视野里，它就是一般聊天。"
        }
    },
    {
        "input": "算了，今天太累了不想听歌了，晚安",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "明确终止推荐与点播环节并提出下线的告别宣言",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "非常常见的对话结束语。判定为单纯的闲谈，回复一句“晚安好梦”并在底层打平所有指令。"
        }
    }
,
    {
        "input": "这首歌设成我手机的来电铃声吧",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "当前播放曲目"
                ]
            },
            "context": "将当前正在播放的曲目截取片段设置为手机铃声的系统级操作",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'设成铃声'是一种资产占有行为（acquire），需要将音频文件裁剪并注册到系统层面。"
        }
    },
    {
        "input": "把这首歌分享到我的微信朋友圈",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "当前播放曲目"
                ]
            },
            "context": "将当前曲目生成分享卡片或链接发送到社交媒体的外部行为",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "分享到朋友圈属于资产外部传播行为，归类为 acquire_music 的社交分发子分支。"
        }
    },
    {
        "input": "我要把刚才那三首全部加到'跑步歌单'里",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "最近播放的三首曲目"
                ]
            },
            "context": "批量将最近播放的多首歌曲归档到用户自定义的特定主题歌单中",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "批量入库歌单操作，涉及多首曲目的整理归档，属于典型的 acquire 意图。"
        }
    },
    {
        "input": "帮我把邓紫棋的《光年之外》和《泡沫》都下载成无损的",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "邓紫棋 光年之外",
                    "邓紫棋 泡沫"
                ]
            },
            "context": "精确指定歌手和双曲目名进行高品质无损格式音频下载",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "明确了歌手和两首歌名，且下载无损格式是纯粹的资产获取行为。"
        }
    },
    {
        "input": "帮我新建一个歌单叫'深夜emo专用'然后把现在这首加进去",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "当前播放曲目"
                ]
            },
            "context": "创建一个全新的自定义歌单并将当前曲目作为第一首入库",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "新建歌单+入库曲目是一个二合一的资产管理操作，归类为 acquire。"
        }
    },
    {
        "input": "能不能把这首歌的高潮部分截出来做成闹钟铃声",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "当前播放曲目"
                ]
            },
            "context": "对当前播放歌曲进行精细化音频剪辑并设置为闹钟提醒音的系统级资产操作",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'截出来做闹钟铃声'涉及音频裁剪和系统级设置，属于 acquire 里偏高阶的资产加工操作。"
        }
    },
    {
        "input": "这首歌的歌词好棒，帮我保存一下歌词文本到备忘录",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "当前播放曲目"
                ]
            },
            "context": "将当前曲目的歌词文本导出保存到手机备忘录等外部应用中",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "保存歌词文本是一种跨应用的资产导出操作，不涉及任何推荐和播放链路。"
        }
    },
    {
        "input": "我要把正在听的这整张专辑全部下载到手机里面离线听",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "当前播放专辑全部曲目"
                ]
            },
            "context": "将当前播放来源的完整专辑批量下载到本地以支持离线收听",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "整张专辑批量下载是一个大规模的资产离线缓存操作。"
        }
    },
    {
        "input": "按照我以前听歌的口味帮我推几首新的吧",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "基于用户长期积累的听歌行为画像进行个性化新歌推荐",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'以前听歌口味'直接指向用户历史偏好画像，需要调用个人化推荐而非公域检索。"
        }
    },
    {
        "input": "上个月我疯狂单曲循环的那几首给我再来一遍",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "请求回放近一个月内具有极高播放频次的个人热门曲目",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'上个月疯狂单曲循环'指向个人播放历史中高频曲目的回溯播放。"
        }
    },
    {
        "input": "给我推荐跟我歌单里风格差不多但我没听过的歌",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "基于用户已有歌单的音乐风格特征进行相似但未接触过的新歌发现型推荐",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'跟我歌单风格差不多+没听过'二合一，属于从个人偏好出发的新歌探索型推荐。"
        }
    },
    {
        "input": "帮我从所有标了红心的歌里面找几首适合现在开车听的",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "在用户红心收藏库内带有场景筛选条件（开车）的子集精选",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "虽然附带了'开车'场景，但核心数据源是'所有标了红心的'个人收藏。先从偏好库里拉取再做场景二次筛选。"
        }
    },
    {
        "input": "我记得去年冬天下大雪的时候有首歌我循环了一整天，帮我找找",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "根据模糊的时间记忆和天气场景还原用户历史中某首高频播放曲目",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "带有强烈的个人历史属性和时间线索，'去年冬天循环了一整天'必须从用户播放记录中回溯定位。"
        }
    },
    {
        "input": "我那个叫'考研冲刺'的歌单好久没更新了帮我按照里面的风格补几首新的",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "根据用户既有的自建主题歌单内容分析其偏好方向并补充新曲目",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "基于'考研冲刺'歌单的既有内容做风格延伸推荐，数据来源是用户自建歌单而非公域库。"
        }
    },
    {
        "input": "系统你应该知道我的口味吧，别让我选了直接放你觉得我会喜欢的",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "完全信任系统对自身偏好的建模能力并允许系统自主决策的无限授权式推荐",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'你应该知道我的口味'明确要求算法根据已掌握的个人画像自主推荐，极典型的favorites意图。"
        }
    },
    {
        "input": "翻翻我高三那年的听歌记录随便挑几首放放",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "要求回溯特定人生阶段（高三）的听歌历史并随机播放",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'高三那年的听歌记录'是极其明确的个人历史回溯需求，只能通过用户数据工具链实现。"
        }
    },
    {
        "input": "上一首好听多了，切回去",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "上一首播放曲目"
            },
            "context": "对当前曲目不满意并要求切回之前播放的那首歌曲",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'上一首'是典型的播放历史上下文指代，需要从播放队列中回退一首。"
        }
    },
    {
        "input": "刚才你推荐的第二首再放一遍",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "上轮推荐的第二首"
            },
            "context": "引用上一轮推荐列表中的具体序号位置进行重播",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "明确引用了上一轮推荐结果中的'第二首'，属于多轮上下文指代的精确点播。"
        }
    },
    {
        "input": "嗯那个歌手还有别的好听的吗",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "上文提及的歌手"
            },
            "context": "基于对话上文中最近提及的某位歌手发起的延伸推荐请求",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'那个歌手'是多轮上下文代词指代，需要从对话历史中解析出具体的歌手实体。"
        }
    },
    {
        "input": "换一首，这首不太行",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "放松"
            },
            "context": "对当前播放曲目表达不满并请求系统更换推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'换一首'隐含重新推荐的诉求，由于没有给出额外约束，默认以放松为兜底情绪重新随机推荐。"
        }
    },
    {
        "input": "跟这首差不多感觉的多来几首",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "当前曲目情绪"
            },
            "context": "以当前正在播放的曲目为锚点寻找听感相似的一批歌曲",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'跟这首差不多感觉'需要从当前播放曲目中提取声学特征作为向量锚点进行相似度召回。"
        }
    },
    {
        "input": "那个...你前面放的那首粤语歌叫什么名字来着",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "回忆不起之前播放过的某首粤语歌的名称而向系统求助",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "用户在回忆之前播放的歌名，属于对话历史查询，由大模型根据上下文直接回答即可。"
        }
    },
    {
        "input": "那这首歌是在哪张专辑里面的",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "对当前或刚才讨论的歌曲所属专辑信息的追问",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'这首歌'是上下文指代，问的是专辑归属等百科信息，交给大模型的知识库回答。"
        }
    },
    {
        "input": "继续放类似的就行不用换风格",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "当前曲目情绪"
            },
            "context": "明确表示对当前推荐方向满意并要求系统继续沿用相同风格持续推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'继续放类似的'相当于延续上一轮推荐的所有条件，以当前曲目为锚做同质化推荐。"
        }
    },
    {
        "input": "对对对就是这个味道，这个歌手还有专辑吗全放了",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "当前播放曲目的歌手"
            },
            "context": "对当前推荐曲目极度认可后进一步搜索该歌手的全部专辑作品",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'这个歌手全部专辑'需要从上下文获取歌手实体后做作品全集搜索。"
        }
    },
    {
        "input": "算了还是切回去放刚才那首吧我越听越上头",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "之前播放曲目"
            },
            "context": "反悔切歌决定并要求回退到之前那首正在享受的曲目",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'切回去放刚才那首'是明确的上下文回退操作，指向播放队列中的前一首。"
        }
    },
    {
        "input": "来点chill vibes的华语R&B，周末躺平用的",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "R&B",
                "language": "中文",
                "mood": "放松"
            },
            "context": "使用英文俚语chill vibes表达放松氛围诉求，结合华语R&B流派的周末躺平场景推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "r&b",
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": "Chinese",
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'chill vibes'是英文口语化的'放松氛围'表达，结合华语限定和R&B流派构成典型的中英混搭推荐。"
        }
    },
    {
        "input": "唔想听啲好sad嘅广东歌，最好系张国荣嗰种feel",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "张国荣",
                "mood": "伤感",
                "language": "粤语"
            },
            "context": "完全使用粤语口语发起的基于张国荣风格的伤感粤语歌推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "张国荣",
                    "Leslie Cheung"
                ],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "伤感",
                "graph_language_filter": "Cantonese",
                "graph_region_filter": "Hong Kong",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "纯粤语输入（唔想=不想、啲=一些、嘅=的、嗰种=那种），需要识别出语言为粤语并提取张国荣实体。"
        }
    },
    {
        "input": "放首lo-fi hip hop beats吧，适合coding的那种background music",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "工作/编程",
                "genre": "嘻哈/电子"
            },
            "context": "程序员群体中流行的低保真嘻哈背景音乐需求",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "hip-hop",
                "graph_scenario_filter": "工作",
                "graph_mood_filter": "平静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "全英文亚文化术语lo-fi hip hop beats+coding+background music，映射到工作场景+嘻哈流派+平静情绪。"
        }
    },
    {
        "input": "有冇啲好似陈奕迅early period嘅ballad，要Cantonese版",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "陈奕迅",
                "language": "粤语",
                "mood": "怀旧"
            },
            "context": "粤语+英语高度混搭的查询，要求陈奕迅早期粤语抒情慢歌",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "陈奕迅",
                    "Eason Chan"
                ],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "怀旧",
                "graph_language_filter": "Cantonese",
                "graph_region_filter": "Hong Kong",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "粤英极度混搭（'有冇啲'、'好似'=粤语；'early period'、'ballad'、'Cantonese'=英文），识别陈奕迅+粤语+怀旧。"
        }
    },
    {
        "input": "想听Taylor Swift那种country pop但是翻唱成中文版的有没有",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "Taylor Swift",
                "genre": "乡村流行",
                "language": "中文"
            },
            "context": "要求寻找泰勒·斯威夫特乡村流行作品的中文翻唱版本",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "Taylor Swift",
                    "泰勒·斯威夫特"
                ],
                "graph_genre_filter": "country",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": "Chinese",
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "Taylor Swift 泰勒 乡村流行 country pop 中文翻唱版"
            },
            "reasoning": "跨语言翻唱搜索极度依赖web_search，本地库不太可能标注'中文翻唱Taylor Swift'这样的元信息。"
        }
    },
    {
        "input": "我要听K-pop girl group嘅歌，BLACKPINK IVE呢啲都得",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "K-pop",
                "region": "韩国"
            },
            "context": "粤语+英文+韩流黑话混搭的韩国女团流行舞曲推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "BLACKPINK",
                    "IVE"
                ],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "激情",
                "graph_language_filter": "Korean",
                "graph_region_filter": "Korea",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "三语混搭（'嘅'=粤语、'K-pop girl group'=英文、'BLACKPINK IVE'=韩团实体）。虽然列举了女团名，但核心诉求是'K-pop女团歌'的流派推荐。"
        }
    },
    {
        "input": "最近好中意听City Pop，有冇啲80年代日本嘅推荐",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "流行",
                "region": "日本",
                "mood": "怀旧"
            },
            "context": "粤语+英文混搭请求80年代日本都会流行乐（City Pop）",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "怀旧",
                "graph_language_filter": "Japanese",
                "graph_region_filter": "Japan",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "粤语'好中意'(很喜欢)+'有冇啲'(有没有)+英文'City Pop'+限定条件'80年代日本'。映射到Japan/Japanese+pop+怀旧。"
        }
    },
    {
        "input": "有没有那种verse是中文但chorus用英文唱的双语歌曲",
        "output": {
            "intent_type": "search",
            "parameters": {
                "language": "中英双语"
            },
            "context": "搜索具有特殊语言编排结构（主歌中文+副歌英文）的双语歌曲",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "中英双语歌曲 主歌中文 副歌英文 verse Chinese chorus English"
            },
            "reasoning": "使用了音乐专业术语verse/chorus的中英混搭表达，搜索双语切换歌曲需要web_search辅助。"
        }
    },
    {
        "input": "我想listen to some Mandarin cover of日本动漫OP，最好是热血番的那种",
        "output": {
            "intent_type": "search",
            "parameters": {
                "language": "中文",
                "genre": "动漫/ACG",
                "mood": "激昂"
            },
            "context": "中英日三语混搭搜索日本热血动漫片头曲的中文翻唱版",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "激昂",
                "graph_language_filter": "Chinese",
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "日本动漫 热血番 OP 片头曲 中文翻唱 Mandarin cover"
            },
            "reasoning": "极度混搭输入（'listen to'英文、'Mandarin cover of'英文、'日本动漫OP'中日交叉）。翻唱类内容依赖web_search。"
        }
    },
    {
        "input": "嚟首beyond嘅《海阔天空》粤语原版，唔好畀国语版我",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "海阔天空",
                "artist_name": "Beyond"
            },
            "context": "纯粤语口语指定播放Beyond的经典粤语原版歌曲并排斥国语版本",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "纯粤语输入（'嚟首'=来首，'唔好畀'=不要给），但歌手+歌名完全明确（Beyond/海阔天空），精确点播。"
        }
    },
    {
        "input": "想要vibe check一下，推荐几首aesthetic感爆棚的dreampop",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "梦幻流行",
                "mood": "梦幻"
            },
            "context": "使用纯Z世代英文网络黑话结合音乐子流派术语的高审美推荐请求",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "indie",
                "graph_scenario_filter": None,
                "graph_mood_filter": "梦幻",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'vibe check'、'aesthetic'、'dreampop'全是英文亚文化用语。dreampop归类为indie子流派+梦幻情绪标签。"
        }
    },
    {
        "input": "我想听嗰啲好old school嘅粤语funk，类似太极乐队嗰种",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "太极乐队",
                "genre": "放克",
                "language": "粤语"
            },
            "context": "粤英混搭请求以太极乐队为参考的80-90年代老派粤语放克音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "太极乐队",
                    "Tai Chi"
                ],
                "graph_genre_filter": "funk",
                "graph_scenario_filter": None,
                "graph_mood_filter": "怀旧",
                "graph_language_filter": "Cantonese",
                "graph_region_filter": "Hong Kong",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "粤英混搭（'嗰啲好old school嘅'=那些很Old School的），识别太极乐队实体+粤语+放克+怀旧。"
        }
    },
    {
        "input": "不要男声，给我推纯女声的清新民谣",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "民谣",
                "mood": "治愈"
            },
            "context": "明确排斥男性歌手人声并限定女声清新质感的民谣推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "folk",
                "graph_scenario_filter": None,
                "graph_mood_filter": "治愈",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'不要男声+纯女声'是性别维度否定，需要结合向量做人声性别识别加上folk+治愈的多维推荐。"
        }
    },
    {
        "input": "换一批，刚才那些都太老了我不喜欢",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "放松"
            },
            "context": "对上一轮推荐结果全面否定（太老）并要求更换为更新潮的曲目",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'换一批'表示全面否定上轮结果，'太老了'暗示需要更新的音乐。由于无其他约束，兜底为放松情绪重发。"
        }
    },
    {
        "input": "除了周杰伦和林俊杰以外的华语男歌手有什么好推荐的",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "流行",
                "region": "华语",
                "mood": "开心"
            },
            "context": "明确排除两位顶流歌手后寻找其他华语男歌手的优质作品",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "周杰伦",
                    "Jay Chou",
                    "林俊杰",
                    "JJ Lin"
                ],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": "Chinese",
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "双实体反向排除（周杰伦+林俊杰作为负面锚点），图谱需标记这两人用于结果剔除，同时限定华语男歌手范围。"
        }
    },
    {
        "input": "能不能不推这种电子合成器的声音，听着脑壳疼，来点纯原声乐器的",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "民谣/原声",
                "mood": "平静"
            },
            "context": "强烈排斥电子合成音色后转向要求纯原声乐器（吉他、钢琴等）的有机音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "folk",
                "graph_scenario_filter": None,
                "graph_mood_filter": "平静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "强烈杜绝electronic的否定逻辑，反向锁定acoustic/folk类的原声器乐，向量负责甄别纯器乐质感。"
        }
    },
    {
        "input": "什么都行就是千万别推抖音神曲那种口水歌给我",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "放松"
            },
            "context": "排斥低质量网络流行口水歌后的无具体方向泛推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'什么都行'说明没有正向约束，核心是负向剔除（不要抖音口水歌），系统需要通过向量避免推出低口碑的洗脑神曲。"
        }
    },
    {
        "input": "不要英文歌不要日语歌不要韩语歌我只听中文",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "放松",
                "language": "中文"
            },
            "context": "通过连续三次否定非中文语种来表达极其强烈的中文歌曲偏好",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": "Chinese",
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "连续三重否定（不要英/日/韩）本质上就是在正向限定Chinese语种，直接锁死语言过滤器即可。"
        }
    },
    {
        "input": "我受不了这种慢悠悠的节奏了，来点快的猛的",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "激昂"
            },
            "context": "因长期听慢歌导致的审美疲劳爆发，要求立刻转向高能快节奏音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "激昂",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "否定了慢节奏（'受不了慢悠悠'），正向要求'快的猛的'直接映射为激昂/激情情绪驱动推荐。"
        }
    },
    {
        "input": "别给我推那种苦情歌了行不行，我已经走出来了要正能量的",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "开心/治愈"
            },
            "context": "明确拒绝伤感情歌类型并要求转向积极正面情绪的音乐推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "开心",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "否定了伤感/苦情歌，'已经走出来了+正能量'强烈暗示情绪已从低谷反弹，需要开心/治愈型歌曲巩固。"
        }
    },
    {
        "input": "除了古典乐和爵士以外什么流派都可以接受",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "放松"
            },
            "context": "通过排除两个特定流派来框定其余所有流派均可接受的宽容泛推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "双流派否定（not classical, not jazz）但没有给出正向流派。系统需从向量层面避开古典和爵士两大类，其余随机推荐。"
        }
    },
    {
        "input": "不要给我推荐任何带鼓点的歌，我头炸了就想听纯钢琴",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "古典/纯音乐",
                "mood": "平静"
            },
            "context": "极度生理排斥打击乐器后转向要求仅含钢琴独奏的纯器乐作品",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "classical",
                "graph_scenario_filter": None,
                "graph_mood_filter": "平静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "否定维度是乐器层面（排斥鼓点），正向需纯钢琴，映射为classical+平静，由向量严格筛选无打击乐的纯钢琴音轨。"
        }
    },
    {
        "input": "我不想再听华语歌了好腻，有没有什么小众的北欧独立乐队推荐",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "独立",
                "region": "北欧"
            },
            "context": "因华语歌曲审美疲劳而转向探索北欧小众独立乐队音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "indie",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": "Western",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "北欧 indie 独立乐队 小众 瑞典 挪威 冰岛 推荐"
            },
            "reasoning": "否定华语+正向北欧独立。'小众北欧独立乐队'本地库可能覆盖不足，启用web_search拓宽发现面。"
        }
    },
    {
        "input": "别再推陈奕迅了我都听烂了，换个港区歌手来",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "陈奕迅",
                "region": "香港地区"
            },
            "context": "明确排斥过度推荐的陈奕迅并要求更换为其他香港歌手",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "陈奕迅",
                    "Eason Chan"
                ],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": "Cantonese",
                "graph_region_filter": "Hong Kong",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'别再推陈奕迅'是负面实体排除，但限定了'港区歌手'的正向地域约束。图谱标记陈奕迅为剔除对象，同时限定Hong Kong+Cantonese。"
        }
    }
,
    {
        "input": "给我推荐几首适合雨天在咖啡厅看书时听的日语女声爵士",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "阅读/咖啡厅",
                "genre": "爵士",
                "language": "日语",
                "mood": "放松"
            },
            "context": "四维交叉：场景(咖啡厅阅读)+流派(爵士)+语种(日语)+性别音色(女声)",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "jazz",
                "graph_scenario_filter": "学习",
                "graph_mood_filter": "放松",
                "graph_language_filter": "Japanese",
                "graph_region_filter": "Japan",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "四维复合：场景(咖啡厅看书→学习/放松) + 流派(jazz) + 语言(Japanese) + 情绪(放松)。女声由向量声学把控。"
        }
    },
    {
        "input": "找几首90年代的港片动作电影里面那种节奏紧张的粤语配乐，最好带点电吉他riff",
        "output": {
            "intent_type": "search",
            "parameters": {
                "region": "香港地区",
                "language": "粤语",
                "genre": "摇滚/原声",
                "mood": "激昂"
            },
            "context": "五维复合：地域(香港)+年代(90年代)+场景(动作电影)+流派(摇滚配乐)+乐器(电吉他)",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "rock",
                "graph_scenario_filter": None,
                "graph_mood_filter": "激昂",
                "graph_language_filter": "Cantonese",
                "graph_region_filter": "Hong Kong",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "90年代 香港动作电影 粤语 配乐 原声带 电吉他 摇滚"
            },
            "reasoning": "港片动作配乐+90年代+电吉他riff过于具体，本地库难以精准匹配，需web_search结合向量做声学筛选。"
        }
    },
    {
        "input": "有没有那种适合秋天傍晚在海边散步时听的、带一点忧郁但又不会太压抑的英文独立民谣",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "散步",
                "mood": "忧郁",
                "genre": "独立民谣",
                "language": "英文"
            },
            "context": "五维交叉：季节氛围(秋天傍晚)+场景(海边散步)+情绪(轻度忧郁)+流派(indie folk)+语种(English)",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "folk",
                "graph_scenario_filter": "散步",
                "graph_mood_filter": "忧郁",
                "graph_language_filter": "English",
                "graph_region_filter": "Western",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "极其丰满的画面感描述。散步+忧郁+folk+English+Western多重叠加，向量负责把控'不会太压抑'的尺度。"
        }
    },
    {
        "input": "推荐适合两个人约会吃法餐时候放的浪漫法语香颂，不要太吵也不要太闷",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "约会/用餐",
                "mood": "浪漫",
                "language": "法语",
                "genre": "香颂"
            },
            "context": "四维组合：场景(约会法餐)+情绪(浪漫)+语种(法语)+流派(香颂)+音量约束(不吵不闷)",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "folk",
                "graph_scenario_filter": "聚会",
                "graph_mood_filter": "浪漫",
                "graph_language_filter": "Others",
                "graph_region_filter": "Western",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "约会法餐场景+浪漫+法语(Others)+香颂(归入folk)。'不太吵不太闷'由向量声学做动态范围筛选。"
        }
    },
    {
        "input": "帮我找适合凌晨三点独自开车穿越隧道时听的、有点赛博朋克味道的日系电子女声",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "开车",
                "genre": "电子",
                "language": "日语",
                "mood": "梦幻"
            },
            "context": "五维极致：时间(凌晨三点)+场景(独自开车隧道)+风格(赛博朋克)+地域流派(日系电子)+性别(女声)",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "electronic",
                "graph_scenario_filter": "开车",
                "graph_mood_filter": "梦幻",
                "graph_language_filter": "Japanese",
                "graph_region_filter": "Japan",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "画面感极强的午夜驾车场景。赛博朋克→electronic+梦幻，日系女声→Japanese+向量筛选。五维叠加。"
        }
    },
    {
        "input": "想要一首既能让我哭出来又能在哭完之后觉得被治愈了的、带着钢琴伴奏的中文慢歌",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "伤感/治愈",
                "genre": "流行",
                "language": "中文"
            },
            "context": "四维+矛盾情绪：情感(先伤感后治愈的双阶段情绪)+乐器(钢琴)+语种(中文)+节奏(慢歌)",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "治愈",
                "graph_language_filter": "Chinese",
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "双阶段情绪需求（先哭后治愈），取治愈作为主标签因为这是用户的终极目的。钢琴+中文+慢歌由向量配合。"
        }
    },
    {
        "input": "有没有那种像宫崎骏电影里久石让配乐风格的、但是是中国作曲家写的、用二胡和古筝演奏的纯音乐",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "久石让",
                "genre": "纯音乐/国风",
                "region": "内地"
            },
            "context": "六维复合：参考人物(久石让)+限定国籍(中国作曲家)+乐器(二胡古筝)+形式(纯音乐)+排除(不要久石让本人)",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "久石让",
                    "Joe Hisaishi"
                ],
                "graph_genre_filter": "classical",
                "graph_scenario_filter": None,
                "graph_mood_filter": "治愈",
                "graph_language_filter": None,
                "graph_region_filter": "Mainland China",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "中国作曲家 久石让风格 宫崎骏 二胡 古筝 纯音乐 治愈"
            },
            "reasoning": "参考锚点是久石让但排除其本人，限定中国作曲家+民族乐器（二胡古筝）。本地难以交叉匹配，需web辅助。"
        }
    },
    {
        "input": "来首2024年Billboard Hot 100上过榜的、带有拉丁节拍但歌词是英文的、女歌手唱的夏天派对歌",
        "output": {
            "intent_type": "search",
            "parameters": {
                "genre": "拉丁/流行",
                "language": "英文",
                "mood": "激情",
                "region": "欧美"
            },
            "context": "六维极限：时效(2024年)+榜单(Billboard)+流派(拉丁节拍)+语种(英文)+性别(女歌手)+场景(夏天派对)",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "latin",
                "graph_scenario_filter": "聚会",
                "graph_mood_filter": "激情",
                "graph_language_filter": "English",
                "graph_region_filter": "Western",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "2024 Billboard Hot 100 拉丁节拍 英文 女歌手 夏天派对歌 Latin pop"
            },
            "reasoning": "六维交叉+时效性(2024 Billboard)强制触发web_search。拉丁节拍+英文+女声+派对+激情多维图向量协同。"
        }
    },
    {
        "input": "我想听韩国producer制作的、给日本偶像团体写的、带有80年代city pop复古感的J-pop舞曲",
        "output": {
            "intent_type": "search",
            "parameters": {
                "genre": "J-pop/City Pop",
                "region": "日本",
                "mood": "怀旧"
            },
            "context": "四维+跨国制作：制作人国籍(韩国)+表演者(日本偶像)+风格(80年代city pop复古)+大类(J-pop舞曲)",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "怀旧",
                "graph_language_filter": "Japanese",
                "graph_region_filter": "Japan",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "韩国 producer 制作 日本偶像 J-pop city pop 复古 80年代 舞曲"
            },
            "reasoning": "跨国产业链查询（韩国制作人+日本偶像），这种幕后制作信息只能靠web_search拿到。"
        }
    },
    {
        "input": "给我整一个专门适合冬天泡温泉时候听的歌单，要有和风纯音乐也要有轻松的英文爵士，交替着来",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "泡温泉/放松",
                "genre": "和风纯音乐+英文爵士",
                "mood": "放松"
            },
            "context": "双流派交替编排的场景化歌单：和风纯音乐与英文爵士的混合播放需求",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "jazz",
                "graph_scenario_filter": "睡前",
                "graph_mood_filter": "放松",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "极其特殊的双流派混编需求。泡温泉→放松/睡前场景，和风+爵士需要系统做两轮不同条件的检索并交替排列。"
        }
    },
    {
        "input": "推荐一些既适合当学习BGM又能在聚会上当背景音乐的、不挑场合的万能纯音乐",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "纯音乐",
                "mood": "放松"
            },
            "context": "跨场景通用型纯音乐推荐，要求同时满足安静学习和社交聚会两种极端场景",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "classical",
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "矛盾场景组合（学习需安静 vs 聚会需氛围），取交集为中等能量的放松纯音乐，由向量精确校准动态范围。"
        }
    },
    {
        "input": "有没有那种歌词里面提到了星星月亮宇宙这些意象的、带点太空迷幻感的Progressive Rock",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "前卫摇滚",
                "mood": "梦幻"
            },
            "context": "四维组合：歌词意象(天体/宇宙)+风格(太空迷幻)+流派(Progressive Rock)+情绪(梦幻)",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "rock",
                "graph_scenario_filter": None,
                "graph_mood_filter": "梦幻",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "歌词意象（星星月亮宇宙）+太空迷幻+前卫摇滚。标签层面rock+梦幻，歌词语义匹配完全靠向量深度理解。"
        }
    },
    {
        "input": "我老婆怀孕了，推荐适合胎教的、莫扎特效应那种古典钢琴，但不要太严肃的，要温馨可爱一点的",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "胎教",
                "genre": "古典",
                "mood": "温馨"
            },
            "context": "五维：场景(胎教)+参考理论(莫扎特效应)+流派(古典钢琴)+情绪(温馨可爱)+否定(不要太严肃)",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "莫扎特",
                    "Mozart"
                ],
                "graph_genre_filter": "classical",
                "graph_scenario_filter": None,
                "graph_mood_filter": "治愈",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "胎教场景+莫扎特古典钢琴+温馨治愈。否定'太严肃'意味着排除贝多芬式的激烈交响，向量抓取轻快小奏鸣曲。"
        }
    },
    {
        "input": "给我推点能在重庆火锅店里放、又辣又燥又带劲的四川方言说唱或者重庆本地乐队的歌",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "说唱/摇滚",
                "region": "内地/西南",
                "mood": "激情",
                "language": "方言"
            },
            "context": "五维地域文化：场景(火锅店)+地域(重庆/四川)+语种(方言)+流派(说唱/摇滚)+情绪(激情)",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "hip-hop",
                "graph_scenario_filter": "聚会",
                "graph_mood_filter": "激情",
                "graph_language_filter": "Chinese",
                "graph_region_filter": "Mainland China",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "重庆 四川 方言说唱 hip-hop 本地乐队 火锅 川渝 rapper"
            },
            "reasoning": "极强地域文化标签（川渝方言说唱），本地库可能缺乏方言标注，需web_search辅助发现。"
        }
    },
    {
        "input": "有没有那种周杰伦曲风+方文山歌词风格+蔡依林唱的、中国风但又带电子节拍的跨界合作",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "周杰伦, 方文山, 蔡依林",
                "genre": "中国风/电子"
            },
            "context": "三实体+双流派的极限组合：要求同时满足三位华语顶流的风格特征交叉",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "周杰伦",
                    "Jay Chou",
                    "方文山",
                    "蔡依林",
                    "Jolin Tsai"
                ],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": "Chinese",
                "graph_region_filter": "Taiwan",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "周杰伦 方文山 蔡依林 合作 中国风 电子节拍 跨界"
            },
            "reasoning": "三个重量级实体交叉+中国风+电子节拍，这种特定合作信息强依赖web_search确认是否存在。"
        }
    },
    {
        "input": "不是这首，是再上一首的上一首",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "播放队列中倒数第三首"
            },
            "context": "多层嵌套的上下文回退指代，要求回退到播放历史更早的位置",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'上一首的上一首'是多层嵌套回退，等效于倒退两首到播放队列中更早的位置。"
        }
    },
    {
        "input": "这个歌手的声音我超喜欢，她还跟别人合唱过什么歌吗",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "当前播放曲目的歌手"
            },
            "context": "基于当前播放歌手发起的合作作品搜索",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": ""
            },
            "reasoning": "'这个歌手'是上下文指代当前播放的歌手，'跟别人合唱'需要搜索合作曲目信息，可能需web辅助。"
        }
    },
    {
        "input": "你说的那个乐队我搜不到，能不能把名字拼写发给我",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "对上一轮AI推荐中提到的某乐队名称请求确认拼写",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "不是搜歌也不是推荐，而是对之前AI回复内容的信息确认请求，交由大模型回答。"
        }
    },
    {
        "input": "那首歌的MV在哪里可以看",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "对上文讨论到的某首歌的MV观看渠道发起追问",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'那首歌'是上下文指代，问的是MV观看平台等信息，属于大模型知识范畴的闲聊。"
        }
    },
    {
        "input": "刚才推的那批里有一首开头是钢琴的，就那首再来一遍",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "上轮推荐中开头为钢琴的曲目"
            },
            "context": "通过音乐特征（钢琴开头）从上一轮推荐列表中定位具体歌曲并重播",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "通过声学特征（开头是钢琴）从上轮推荐中定位，属于多轮上下文+声学特征的交叉指代。"
        }
    },
    {
        "input": "嗯嗯对就是这首，这首歌是几几年出的",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "确认当前或刚才播放的歌曲后追问发行年份信息",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "确认歌曲后追问元数据信息（发行年份），属于百科类闲聊，大模型直接回答。"
        }
    },
    {
        "input": "这种调调的再来五首，但音量大一点的那种",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "当前曲目情绪"
            },
            "context": "要求沿用当前风格但提升音量动态范围的批量推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'这种调调'延续当前风格，'音量大一点'是声学层面的约束，由向量做能量/响度筛选。"
        }
    },
    {
        "input": "你前面说的那个专辑后面几首我也想听听",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "上文提及的歌手"
            },
            "context": "基于之前对话中提到的某张专辑，要求播放该专辑的后续曲目",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'前面说的那个专辑后面几首'需要从对话历史中解析出专辑名，再进行曲目列表搜索。"
        }
    },
    {
        "input": "行了行了别放了，刚才那十首里面帮我挑最好听的三首做个小歌单",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "上轮推荐中的最佳三首"
                ]
            },
            "context": "从上一轮推荐的多首歌曲中由系统筛选最优三首并归档为歌单",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "从上轮推荐结果中精选并创建歌单，是上下文指代+资产管理(acquire)的组合。"
        }
    },
    {
        "input": "对了那个歌手是不是参加过那个什么中国好声音还是歌手来着",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "对上文提及的歌手是否参加过音乐综艺节目的记忆模糊追问",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "模糊的综艺参与历史追问，'那个歌手'需从上下文解析。属于百科类闲聊。"
        }
    },
    {
        "input": "推啲similar to Radiohead嘅experimental rock嘅嘢，要嗰种好creepy嘅",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "Radiohead",
                "genre": "实验摇滚",
                "mood": "诡异"
            },
            "context": "粤英深度混搭：以Radiohead为锚的实验摇滚、要求阴森诡异感",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "Radiohead"
                ],
                "graph_genre_filter": "rock",
                "graph_scenario_filter": None,
                "graph_mood_filter": "忧郁",
                "graph_language_filter": "English",
                "graph_region_filter": "Western",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "粤英混搭（'推啲'=推些、'嘅嘢'=的东西、'嗰種好creepy'=那种很诡异的）。Radiohead+experimental rock+忧郁/诡异。"
        }
    },
    {
        "input": "搵首周杰伦嘅《七里香》嚟听下，好耐冇听过啦",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "七里香",
                "artist_name": "周杰伦"
            },
            "context": "纯粤语口语指定播放周杰伦的经典作品七里香",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "粤语输入（'搵首'=找首、'嚟听下'=来听下、'好耐冇听过'=好久没听过），但歌手+歌名完全明确。"
        }
    },
    {
        "input": "有冇啲好似Adele嗰種powerful vocal但系唱粤语嘅女歌手",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "Adele",
                "language": "粤语"
            },
            "context": "粤英混搭要求以Adele的强力嗓音为参考寻找粤语女歌手",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "Adele",
                    "阿黛尔"
                ],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "伤感",
                "graph_language_filter": "Cantonese",
                "graph_region_filter": "Hong Kong",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "粤英混搭（'有冇啲好似'=有没有类似、'嗰種'=那种、'但系'=但是）。以Adele为vocal参考+限定粤语女歌手。"
        }
    },
    {
        "input": "我想要playlist for late night study session，中文英文混着来都ok",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "学习",
                "language": "中英混合",
                "mood": "平静"
            },
            "context": "中英混搭的深夜学习背景音乐请求，不限语种",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "学习",
                "graph_mood_filter": "平静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'playlist for late night study session'全英文+中文补充说明。学习场景+平静情绪，不限语种。"
        }
    },
    {
        "input": "帮我find几首Mandopop里面有rap verse嘅歌，即系嗰种半唱半说嘅",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "华语流行/说唱",
                "language": "中文"
            },
            "context": "粤英中三语混搭搜索带有说唱段落的华语流行歌",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": "Chinese",
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "极度混搭（'find'英文、'Mandopop'英文术语、'rap verse'英文、'嘅歌'粤语、'即系嗰種'粤语）。华语流行+含说唱段落。"
        }
    },
    {
        "input": "想听嗰啲90s Cantopop classics，梅艳芳王菲嗰个年代嘅",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "粤语流行",
                "region": "香港地区",
                "mood": "怀旧"
            },
            "context": "粤英混搭怀旧90年代港乐黄金时期经典粤语流行",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "梅艳芳",
                    "Anita Mui",
                    "王菲",
                    "Faye Wong"
                ],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "怀旧",
                "graph_language_filter": "Cantonese",
                "graph_region_filter": "Hong Kong",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "粤英混搭（'嗰啲'=那些、'90s Cantopop classics'英文、'嗰个年代嘅'粤语）。梅艳芳王菲作引子实体+90年代+怀旧粤语流行。"
        }
    },
    {
        "input": "gimme some bass-heavy trap beats，中文的那种，类似Higher Brothers的",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "Higher Brothers",
                "genre": "trap/说唱",
                "language": "中文"
            },
            "context": "全英口语+中文补充的中文trap音乐请求",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "Higher Brothers",
                    "海尔兄弟"
                ],
                "graph_genre_filter": "hip-hop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "激情",
                "graph_language_filter": "Chinese",
                "graph_region_filter": "Mainland China",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "口语化英文'gimme some bass-heavy trap beats'+中文限定+Higher Brothers实体。Trap归入hip-hop+激情+Chinese。"
        }
    },
    {
        "input": "有无一啲好jazzy嘅Cantonese cover，将英文jazz standard用广东话唱嘅",
        "output": {
            "intent_type": "search",
            "parameters": {
                "genre": "爵士",
                "language": "粤语"
            },
            "context": "粤英混搭搜索将英文爵士标准曲目翻唱为粤语版本的作品",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "jazz",
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": "Cantonese",
                "graph_region_filter": "Hong Kong",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "粤语翻唱 爵士 jazz standard Cantonese cover 广东话"
            },
            "reasoning": "粤英混搭（'有无一啲好jazzy嘅'、'将英文jazz standard用广东话唱嘅'）。跨语言翻唱需web_search辅助。"
        }
    },
    {
        "input": "play me something by五月天but the acoustic live version，唔好studio版",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "五月天"
            },
            "context": "中英粤三语混搭搜索五月天的不插电现场版本，排斥录音室版",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "五月天",
                    "Mayday"
                ],
                "graph_genre_filter": "rock",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": "Chinese",
                "graph_region_filter": "Taiwan",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "五月天 Mayday acoustic live version 不插电 现场版"
            },
            "reasoning": "三语混搭（'play me'英文、'五月天'中文、'唔好'粤语）。acoustic live版本需要web_search定位特殊版本。"
        }
    },
    {
        "input": "I want to听一些很有氛围感的ambient electronic，适合冥想meditation嗰种",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "冥想",
                "genre": "电子/氛围",
                "mood": "平静"
            },
            "context": "中英粤混搭的冥想氛围电子音乐请求",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "electronic",
                "graph_scenario_filter": "睡前",
                "graph_mood_filter": "平静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "三语混搭（'I want to听'中英、'ambient electronic'英文、'meditation嗰种'粤英混合）。冥想→睡前/平静+electronic。"
        }
    },
    {
        "input": "搵首好sweet嘅love song，最好系男女duet嗰種，Mandarin或者Cantonese都得",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "浪漫",
                "language": "中文/粤语"
            },
            "context": "粤英中混搭搜索甜蜜男女对唱情歌",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "浪漫",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "粤英混搭（'搵首好sweet嘅'=找首很甜的、'男女duet嗰種'=男女对唱那种、'Mandarin或者Cantonese都得'）。浪漫情歌+不限中粤语。"
        }
    },
    {
        "input": "any recommendations for workout music？要嗰種好hype嗰種，Cantonese hip-hop有冇",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "健身",
                "genre": "嘻哈",
                "language": "粤语",
                "mood": "激昂"
            },
            "context": "粤英混搭的健身粤语嘻哈音乐推荐请求",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "hip-hop",
                "graph_scenario_filter": "健身",
                "graph_mood_filter": "激昂",
                "graph_language_filter": "Cantonese",
                "graph_region_filter": "Hong Kong",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "粤语 hip-hop 说唱 广东话 健身 workout hype"
            },
            "reasoning": "粤英混搭（'any recommendations for workout music'英文、'好hype嗰種'粤英、'有冇'粤语）。粤语hip-hop较小众需web辅助。"
        }
    },
    {
        "input": "呢首歌好好听呀，佢嘅其他album有冇得download offline",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "当前播放歌手的其他专辑"
                ]
            },
            "context": "纯粤语表达对当前歌手其他专辑的离线下载需求",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "粤英混搭（'呢首歌好好听呀'=这首歌很好听啊、'佢嘅'=他的、'有冇得download offline'=能不能离线下载）。资产下载意图。"
        }
    },
    {
        "input": "我嗰个叫'Chill Vibes'嘅playlist可唔可以share畀我朋友听",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "Chill Vibes歌单"
                ]
            },
            "context": "粤英混搭请求分享个人歌单给朋友",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "粤英混搭（'我嗰个'=我那个、'可唔可以share畀'=能不能分享给）。歌单分享属于资产社交传播。"
        }
    },
    {
        "input": "不要推荐任何翻唱版本，我只要原唱原版",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "放松"
            },
            "context": "严格排除翻唱版本仅接受原唱的版本纯净度要求",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "版本层面的否定约束（不要翻唱），系统需要在结果中标注版本类型并过滤。无其他正向约束，默认放松。"
        }
    },
    {
        "input": "能不能别推荐小鲜肉偶像的歌了，给我整点老炮儿级别的",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "怀旧"
            },
            "context": "排斥年轻流量偶像歌手后转向要求资深实力派歌手作品",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "怀旧",
                "graph_language_filter": "Chinese",
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'小鲜肉偶像'是负向排除（排斥流量明星），'老炮儿级别'要求资深实力派，映射为怀旧+华语。"
        }
    },
    {
        "input": "我讨厌那种千篇一律的4536251和弦走向的流行歌，来点和弦不走寻常路的",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "独立/实验",
                "mood": "平静"
            },
            "context": "从乐理层面排斥常规和弦走向后渴求和声编排新颖独特的音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "indie",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "极其专业的乐理维度否定（排斥4536251套路和弦）。'不走寻常路'指向indie/实验类，向量做和声多样性筛选。"
        }
    },
    {
        "input": "够了够了太吵了！赶紧给我换成安安静静的不要任何鼓点和贝斯",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "平静"
            },
            "context": "因受到高强度音量轰炸后的紧急情绪切换，要求极致安静的纯旋律音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "classical",
                "graph_scenario_filter": None,
                "graph_mood_filter": "平静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "紧急的体感排斥（太吵了！），双乐器否定（排除鼓点+贝斯），正向要求安静→classical+平静+向量做乐器过滤。"
        }
    },
    {
        "input": "别推周深了求求你，他的声音太辨识度了听太多会腻",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "放松"
            },
            "context": "对特定歌手的审美疲劳导致的单实体排除请求",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "周深",
                    "Zhou Shen"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": "Chinese",
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "单实体排除（周深），标记为负面锚点。无正向约束所以兜底为放松。"
        }
    },
    {
        "input": "我不要听任何伤感的歌了这段时间听太多了，但也别给我太嗨的怕吵到室友",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "治愈"
            },
            "context": "同时否定两个极端情绪（不要伤感+不要太嗨）后寻求温和中间带的音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "治愈",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "双极端否定（NOT伤感 AND NOT激情），取中间值'治愈/放松'作为折中情绪。向量把控动态范围不过嗨。"
        }
    },
    {
        "input": "这首歌除了旋律好以外歌词写得太烂了，有没有旋律差不多但歌词有深度的替代品",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "当前曲目情绪"
            },
            "context": "对当前歌曲的旋律满意但歌词不满意，要求保持旋律风格的前提下寻找歌词更优的替代",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "部分否定（旋律OK+歌词差），保留声学相似度搜索但加入歌词质量约束。向量做旋律匹配，歌词深度难以自动判别。"
        }
    },
    {
        "input": "太甜了太甜了受不了了，来点又丧又酷的",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "忧郁"
            },
            "context": "对过度甜蜜音乐的审美疲劳引发的情绪反转到丧酷风格",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "忧郁",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "否定了甜蜜/可爱风格（太甜了），反向要求'丧+酷'→忧郁/独立风格调性转变。"
        }
    },
    {
        "input": "拜托别再放纯音乐了我快睡着了，给我来有人唱歌的",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "开心"
            },
            "context": "因纯音乐导致困倦而要求切换为有人声演唱的歌曲",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "开心",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "否定纯音乐（instrumental）+正向要求有人声。'快睡着了'暗示需要提振精神→开心/流行有声歌曲。"
        }
    },
    {
        "input": "不要再给我推周杰伦林俊杰陈奕迅了好吗，这三个人的歌我全部听了上千遍了",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "放松"
            },
            "context": "同时排除三位华语顶流歌手后的开放式推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "周杰伦",
                    "Jay Chou",
                    "林俊杰",
                    "JJ Lin",
                    "陈奕迅",
                    "Eason Chan"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": "Chinese",
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "三重实体排除（周杰伦+林俊杰+陈奕迅），图谱标记所有实体为负面锚点。无正向约束，兜底放松华语推荐。"
        }
    }
,
    {
        "input": "这首歌太好听了能不能帮我把它设成微信来电铃声",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "当前播放曲目"
                ]
            },
            "context": "将当前正在播放的歌曲设置为微信应用内来电铃声",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "设置应用铃声是资产占有行为，需要音频裁剪并注入应用设置。"
        }
    },
    {
        "input": "帮我把今天收藏的所有歌导出成一个文件发到我邮箱",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "今日收藏全部曲目"
                ]
            },
            "context": "将当天收藏的全部歌曲导出为可传输文件并通过邮件发送",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "批量数据导出+外部渠道发送，属于acquire_music中偏向数据管理的操作。"
        }
    },
    {
        "input": "我想把这个歌单复制一份出来重新改个名字叫'摆烂专用'",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "当前激活歌单"
                ]
            },
            "context": "复制当前歌单并重命名为新名称的歌单管理操作",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "复制歌单+重命名属于资产克隆操作，不涉及推荐或播放。"
        }
    },
    {
        "input": "帮我取消收藏这首，加错了",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "当前播放曲目"
                ]
            },
            "context": "撤销对当前播放歌曲的收藏操作",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "取消收藏是资产管理中的逆向操作（移出收藏库），仍属acquire范畴。"
        }
    },
    {
        "input": "把正在听的这首歌和上一首一起添加到'通勤路上'那个歌单里",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "当前播放曲目",
                    "上一首播放曲目"
                ]
            },
            "context": "将当前和上一首两首歌同时加入到指定主题歌单中",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "多曲目批量入库到指定歌单，涉及上下文指代（上一首）+资产管理双重逻辑。"
        }
    },
    {
        "input": "能不能把这首歌的前奏截出来当我的QQ提示音",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "当前播放曲目"
                ]
            },
            "context": "裁剪当前曲目的前奏片段并设置为社交应用提示音",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "音频精确裁剪（前奏）+设系统提示音，属于高级资产加工操作。"
        }
    },
    {
        "input": "我要清空'夜跑专用'这个歌单然后重新整理",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "夜跑专用歌单"
                ]
            },
            "context": "清空指定歌单所有内容以便重新整理组织",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "清空歌单是资产管理中的批量删除操作，在acquire_music大类下的歌单维护分支。"
        }
    },
    {
        "input": "我好像很久没发现让我惊喜的新歌了，根据我的听歌习惯帮我挖掘一些冷门的",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "基于个人画像的冷门新歌探索，解决用户音乐发现疲劳",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'根据我的听歌习惯+冷门'是在个人偏好画像基础上做长尾探索，典型的favorites意图。"
        }
    },
    {
        "input": "去年我年度播放量最高的那十首歌帮我列出来",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "请求生成用户去年播放频次Top10的个人年度榜单",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "个人年度播放榜单是纯用户数据驱动的回溯操作，只能从个人播放历史中统计。"
        }
    },
    {
        "input": "你分析一下我一般几点听歌最多然后那个时间段给我自动推",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "要求系统分析用户播放行为的时间分布规律并在高频时段自动推荐",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'分析我几点听歌最多'完全是个人行为数据分析+自动化推荐调度，核心依赖用户画像。"
        }
    },
    {
        "input": "帮我看看我跟我女朋友的听歌品味有多少重合的",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "双用户画像对比分析，计算两个人的音乐偏好重叠度",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "双用户偏好对比是个人数据分析的延伸场景，涉及两份用户画像的交集计算。"
        }
    },
    {
        "input": "最近半年我是不是越来越多听民谣了？感觉口味在变",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "用户请求系统分析自身近期流派偏好变化趋势",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "听歌口味趋势分析需要纵向对比用户历史数据，属于favorites意图中的个人洞察分支。"
        }
    },
    {
        "input": "我跟我最好的朋友都喜欢听的歌里有没有我们两个都没听过的好歌",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "基于两个用户共同偏好的交集做新歌发现推荐",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "双人偏好交集→新歌推荐，涉及两个用户画像的融合分析后做探索性推荐。"
        }
    },
    {
        "input": "我今年加了好多歌到收藏里但很多都没再听过了，帮我清理一下不太喜欢的",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "请求系统基于播放频率分析筛选出收藏库中低播放量的歌曲用于清理",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "收藏库清理需要分析每首歌的播放频率和用户互动度，属于个人数据分析+资产管理的交叉，以favorites为主。"
        }
    },
    {
        "input": "找一首既能当婚礼入场BGM、又适合新娘在化妆间准备时循环播放的、带弦乐四重奏的英文歌",
        "output": {
            "intent_type": "search",
            "parameters": {
                "genre": "古典/流行cross",
                "language": "英文",
                "mood": "浪漫"
            },
            "context": "双场景(婚礼+化妆)+乐器(弦乐四重奏)+语种(英文)+情绪(浪漫)的极致组合",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": "聚会",
                "graph_mood_filter": "浪漫",
                "graph_language_filter": "English",
                "graph_region_filter": "Western",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "婚礼入场BGM 弦乐四重奏 英文 浪漫 wedding ceremony string quartet"
            },
            "reasoning": "婚礼场景+弦乐四重奏+英文+浪漫，非常具体的仪式感需求，web_search才能精准定位婚礼BGM类别。"
        }
    },
    {
        "input": "有没有那种在东京涩谷十字路口、下着小雨、戴着耳机走过人群时听的日系都会感R&B",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "R&B",
                "region": "日本",
                "mood": "孤独",
                "language": "日语"
            },
            "context": "极致画面感五维：地标(东京涩谷)+天气(小雨)+场景(步行)+情绪(孤独都会感)+流派(日系R&B)",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "r&b",
                "graph_scenario_filter": "散步",
                "graph_mood_filter": "孤独",
                "graph_language_filter": "Japanese",
                "graph_region_filter": "Japan",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "意象极强的都会孤独画面。涩谷+小雨→散步+孤独，日系R&B→r&b+Japanese+Japan。向量抓取都会夜感。"
        }
    },
    {
        "input": "推荐一些适合在高铁上看窗外风景飞速后退时候听的、既壮阔又有点小伤感的后摇滚",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "旅途/交通",
                "genre": "后摇滚",
                "mood": "壮阔/伤感"
            },
            "context": "四维复合：场景(高铁旅途)+视觉(窗外风景)+情绪(壮阔+微伤)+流派(后摇)",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "rock",
                "graph_scenario_filter": "散步",
                "graph_mood_filter": "伤感",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "高铁旅途的运动感+后摇的壮阔声场+微伤情绪。场景近似散步，伤感取主情绪标签，向量做声场宏大度筛选。"
        }
    },
    {
        "input": "帮我找一些七八十年代黑人灵魂乐女歌手唱的、适合周日早上在厨房做brunch时放的温暖歌曲",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "做饭",
                "genre": "灵魂乐",
                "mood": "温馨",
                "language": "英文"
            },
            "context": "五维极限：年代(70-80年代)+族裔(黑人)+流派(灵魂乐)+性别(女)+场景(厨房做brunch)+情绪(温暖)",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "soul",
                "graph_scenario_filter": "做饭",
                "graph_mood_filter": "治愈",
                "graph_language_filter": "English",
                "graph_region_filter": "Western",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "极致生活画面感。灵魂乐+做饭场景+温暖(治愈)+English+女声由向量甄别。年代限定由向量发行年份配合。"
        }
    },
    {
        "input": "想要一首能让我在跑最后一公里冲刺时爆发出全部力量的、BPM在170以上的硬核电子",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "跑步/冲刺",
                "genre": "硬核电子",
                "mood": "激昂"
            },
            "context": "四维极限：场景(跑步冲刺)+BPM精确要求(170+)+流派(硬核电子)+情绪(极限爆发)",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "electronic",
                "graph_scenario_filter": "健身",
                "graph_mood_filter": "激昂",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "BPM精确约束(170+)是极其具体的声学特征需求。健身冲刺+electronic+激昂，向量做BPM范围筛选。"
        }
    },
    {
        "input": "有没有那种适合一边泡澡一边喝红酒一边看窗外夜景时听的、有点颓废又很高级的法式电子",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "法式电子",
                "mood": "颓废/高级",
                "region": "法国"
            },
            "context": "六维极致生活美学：场景(泡澡)+伴随(红酒)+视觉(夜景)+情绪(颓废高级)+流派(法式电子)+地域(法国)",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "electronic",
                "graph_scenario_filter": "睡前",
                "graph_mood_filter": "孤独",
                "graph_language_filter": "Others",
                "graph_region_filter": "Western",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "极致奢华场景画面。法式电子→electronic+Western+Others。泡澡红酒夜景→睡前+孤独/颓废质感由向量把控。"
        }
    },
    {
        "input": "推荐一些适合在大雪纷飞的夜晚、一个人坐在靠窗的位置、手捧热可可、看着窗外发呆时听的钢琴曲",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "孤独/温暖",
                "genre": "钢琴曲"
            },
            "context": "极致画面感：天气(大雪)+时间(夜晚)+场景(靠窗独处)+道具(热可可)+动作(发呆)+流派(钢琴曲)",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "classical",
                "graph_scenario_filter": "睡前",
                "graph_mood_filter": "孤独",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "极度场景化的孤独温暖矛盾体。大雪+独处→孤独，热可可+发呆→平静/温暖。钢琴曲→classical。向量抓温暖质感的独奏。"
        }
    },
    {
        "input": "找几首同时被纳入了Pitchfork年度最佳和Grammy提名的、2023年发行的独立摇滚专辑里的主打曲",
        "output": {
            "intent_type": "search",
            "parameters": {
                "genre": "独立摇滚",
                "mood": "激昂"
            },
            "context": "六维极限：双榜单(Pitchfork+Grammy)+年份(2023)+流派(indie rock)+专辑地位(主打曲)+行业认可维度",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "indie",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": "English",
                "graph_region_filter": "Western",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "Pitchfork 2023 best album Grammy nominated indie rock lead single 年度最佳"
            },
            "reasoning": "双榜单交叉+年份限定的极端专业查询，只有web_search能对接这种行业元数据。"
        }
    },
    {
        "input": "有没有适合拍vlog当背景音乐的、版权免费的、轻快可爱的ukulele伴奏小品",
        "output": {
            "intent_type": "search",
            "parameters": {
                "genre": "轻音乐/ukulele",
                "mood": "开心"
            },
            "context": "四维功能性需求：用途(vlog BGM)+版权(免费)+情绪(轻快可爱)+乐器(ukulele)",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "folk",
                "graph_scenario_filter": None,
                "graph_mood_filter": "开心",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "版权免费 royalty free ukulele BGM vlog 轻快可爱 背景音乐 creative commons"
            },
            "reasoning": "'版权免费'是极为特殊的法律约束，需要web_search搜索CC协议或royalty-free音乐平台。"
        }
    },
    {
        "input": "想听2000年左右的台湾偶像剧OST，就那种一听到旋律就想起流星花园那个年代的",
        "output": {
            "intent_type": "search",
            "parameters": {
                "region": "台湾地区",
                "genre": "流行/OST",
                "mood": "怀旧"
            },
            "context": "四维怀旧：年代(2000年)+地域(台湾)+类型(偶像剧OST)+参考作品(流星花园)+情绪(怀旧)",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "流星花园"
                ],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "怀旧",
                "graph_language_filter": "Chinese",
                "graph_region_filter": "Taiwan",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "2000年代 台湾偶像剧 OST 原声带 流星花园 薰衣草 斗鱼 恶作剧之吻"
            },
            "reasoning": "年代+地域+类型(偶像剧OST)的高度具体查询。web_search辅助定位年代久远的剧集原声带。"
        }
    },
    {
        "input": "帮我弄一个从早上起床到晚上睡觉按时间线排列的全天候BGM歌单，早上要元气、中午要放松、下午要专注、晚上要浪漫",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "全天候",
                "mood": "多段式"
            },
            "context": "四段时间线×四种情绪的超大型歌单编排需求（元气→放松→专注→浪漫）",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "超级复合需求：四个时段×四种情绪。系统需要做四轮独立检索（元气/放松/平静/浪漫）并按时间线编排成完整歌单。"
        }
    },
    {
        "input": "来一首歌手是亚洲面孔但在欧美厂牌出道、英文演唱的indie pop，最好还入围过什么音乐节",
        "output": {
            "intent_type": "search",
            "parameters": {
                "genre": "indie pop",
                "language": "英文",
                "region": "欧美"
            },
            "context": "五维交叉+隐含条件：歌手族裔(亚洲)+厂牌(欧美)+语种(英文)+流派(indie pop)+成就(音乐节)",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "indie",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": "English",
                "graph_region_filter": "Western",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "亚裔歌手 欧美厂牌 indie pop 英文 音乐节 Asian artist Western label"
            },
            "reasoning": "亚裔+欧美厂牌+英文+indie pop+音乐节入围，极多隐含约束只能靠web_search做人物背景匹配。"
        }
    },
    {
        "input": "我想听一首歌，这首歌要满足：1.是2020年以后的 2.歌手来自东南亚 3.用英文唱 4.风格是tropical house 5.MV至少有一亿播放",
        "output": {
            "intent_type": "search",
            "parameters": {
                "genre": "tropical house",
                "language": "英文",
                "region": "东南亚"
            },
            "context": "用户用编号列表式精确列出五大约束条件的极限搜索",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "electronic",
                "graph_scenario_filter": None,
                "graph_mood_filter": "开心",
                "graph_language_filter": "English",
                "graph_region_filter": "Western",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "Southeast Asian artist tropical house English 2020 MV 1 billion views 东南亚歌手"
            },
            "reasoning": "五条精确约束以编号形式列出，MV播放量这种元数据只有web_search能验证。"
        }
    },
    {
        "input": "推荐一首既可以当手机等待铃声、又可以作为YouTube视频片头音乐的、5秒就能抓耳的电子jingle",
        "output": {
            "intent_type": "search",
            "parameters": {
                "genre": "电子/jingle",
                "mood": "开心"
            },
            "context": "双用途(电话铃声+视频片头)+时长约束(5秒抓耳)+形式(jingle)的功能性极限需求",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "electronic",
                "graph_scenario_filter": None,
                "graph_mood_filter": "开心",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "电子 jingle 片头音乐 短旋律 铃声 5秒 catchy electronic intro"
            },
            "reasoning": "功能性极限查询。jingle(短旋律)+双用途+时长约束。需web_search搜索音效/jingle平台。"
        }
    },
    {
        "input": "就第三首和第五首不错其他删掉，帮我存着",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "上轮推荐的第三首",
                    "上轮推荐的第五首"
                ]
            },
            "context": "从上轮推荐列表中精确挑选两首保存并淘汰其余",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "序号精确指代（第三首+第五首）+保存操作。多轮上下文+acquire交叉。"
        }
    },
    {
        "input": "刚才那首的原唱是谁啊，我听的这个版本是翻唱吗",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "对刚才播放歌曲的版本归属和原唱信息发起追问",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'刚才那首'是上下文指代，问的是原唱/翻唱的版本归属信息，属于百科类闲聊。"
        }
    },
    {
        "input": "把音量调小一点，然后速度放慢到0.8倍",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "对当前播放的音频进行音量和播放速度的调整",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "播放控制指令（音量/倍速调整），不涉及任何推荐或搜索逻辑。"
        }
    },
    {
        "input": "它跟上一首是同一张专辑里的吗",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "对当前播放和上一首播放的歌曲是否属于同一专辑的比较追问",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "双上下文指代（'它'=当前曲目，'上一首'=前一曲目）+专辑归属比较，属于元数据查询。"
        }
    },
    {
        "input": "你不是说这个歌手出过五张专辑吗，按时间线全放一遍",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "上文提及的歌手"
            },
            "context": "基于之前AI回复中陈述的歌手作品数量信息发起的全作品时间线播放",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": ""
            },
            "reasoning": "引用了之前AI的回复内容（五张专辑），需要从上下文解析歌手后做全作品检索。"
        }
    },
    {
        "input": "等一下，这首歌快进到副歌部分直接从那里开始放",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "播放进度控制指令，要求跳转到当前曲目的副歌段落",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "播放控制指令（快进到副歌），不涉及推荐或搜索。"
        }
    },
    {
        "input": "这个歌手长什么样啊，帅不帅",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "对当前播放歌手的外貌长相发起的纯闲聊好奇",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "纯闲聊八卦（歌手外貌），与音乐推荐完全无关，由大模型社交能力接管。"
        }
    },
    {
        "input": "上一轮推的那些我一首都不喜欢，完全不是我的菜，重新来",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "对上轮全部推荐结果的强烈负反馈后要求系统基于个人画像重新校准推荐",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "全量负反馈('一首都不喜欢')表明上轮推荐与用户画像严重偏离，需要调用favorites重新校准匹配。"
        }
    },
    {
        "input": "我觉得你越来越懂我了哈哈，继续这个路线不要变",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "对系统推荐准确度的正面肯定并要求维持当前推荐策略",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "正面反馈+路线锁定。'越来越懂我'意味着系统画像已校准到位，继续用当前偏好模型推荐。"
        }
    },
    {
        "input": "嗯你刚才说这首歌拿过金曲奖是吧，同一届金曲奖还有哪些歌得奖了",
        "output": {
            "intent_type": "search",
            "parameters": {},
            "context": "基于之前AI回复中提到的金曲奖信息进行同届获奖作品的扩展搜索",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "金曲奖 获奖 得奖歌曲"
            },
            "reasoning": "引用AI之前回复中的金曲奖信息做拓展搜索，具体届次需从对话历史中提取，web_search核实。"
        }
    },
    {
        "input": "🎵🎵🎵",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "用户仅发送了音符emoji没有任何文字表达",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "纯emoji输入没有可解析的语义意图，当作闲聊处理，回复引导用户说出具体需求。"
        }
    },
    {
        "input": "啊啊啊啊啊啊啊啊啊啊啊！！！！！太好听了！！！！！！",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "用户对当前播放曲目极度兴奋的纯情感宣泄表达",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "纯情绪爆发（尖叫式兴奋），没有可执行的音乐意图，当闲聊回应（共鸣用户的喜悦）。"
        }
    },
    {
        "input": "我现在的心情大概就是... 算了说不出来，你随便放吧",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "放松"
            },
            "context": "表达力匮乏状态下将推荐权完全交给系统的放弃式委托",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "用户放弃自我描述（'说不出来'），完全委托系统。无任何约束条件，默认放松兜底。"
        }
    }
,
    {
        "input": "我想要一首适合深夜独自在便利店门口蹲着吃泡面时听的、带点城市孤独感但不至于太惨的华语indie",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "孤独",
                "genre": "独立",
                "language": "中文"
            },
            "context": "极致画面感：场景(便利店门口)+动作(蹲着吃泡面)+情绪(城市孤独但不太惨)+流派(华语独立)",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "indie",
                "graph_scenario_filter": None,
                "graph_mood_filter": "孤独",
                "graph_language_filter": "Chinese",
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "深夜便利店+泡面+蹲着=都市底层孤独画面。indie+Chinese+孤独但不太惨，向量需把控悲伤程度不过重。"
        }
    },
    {
        "input": "帮我找一首适合做毕业论文答辩PPT背景音乐的、显得很专业很学术但又不会喧宾夺主的轻音乐",
        "output": {
            "intent_type": "search",
            "parameters": {
                "genre": "轻音乐",
                "mood": "平静"
            },
            "context": "四维功能性：用途(答辩PPT)+调性(专业学术)+约束(不喧宾夺主)+形式(轻音乐)",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "classical",
                "graph_scenario_filter": "学习",
                "graph_mood_filter": "平静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "答辩 PPT 背景音乐 学术 轻音乐 不喧宾夺主 presentation BGM"
            },
            "reasoning": "学术答辩BGM是功能性极强的场景，需要web_search定位适合演示文稿的低干扰纯音乐。"
        }
    },
    {
        "input": "推荐一些适合在老上海风格的鸡尾酒吧里放的、有点复古爵士又带点现代电子混搭的lounge music",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "爵士/电子/lounge",
                "mood": "浪漫",
                "region": "上海/复古"
            },
            "context": "五维：地域风格(老上海)+场景(鸡尾酒吧)+流派(复古爵士+电子混搭)+子类(lounge music)+情绪(浪漫)",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "jazz",
                "graph_scenario_filter": "聚会",
                "graph_mood_filter": "浪漫",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "老上海复古爵士+电子混搭=cross-genre lounge。爵士为主+聚会场景+浪漫，向量匹配复古+电子交叉质感。"
        }
    },
    {
        "input": "有没有那种在藏区自驾穿越无人区时、看着远处雪山听的、有藏族元素融合在里面的后摇或者氛围音乐",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "后摇/氛围",
                "region": "藏区",
                "mood": "壮阔"
            },
            "context": "六维极限画面：场景(自驾无人区)+地标(雪山)+地域音乐(藏族元素)+流派(后摇/氛围)+情绪(壮阔)+活动(旅行)",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "rock",
                "graph_scenario_filter": "开车",
                "graph_mood_filter": "梦幻",
                "graph_language_filter": None,
                "graph_region_filter": "Mainland China",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "藏区 藏族元素 后摇 氛围音乐 世界音乐 雪山 公路 ambient post-rock Tibet"
            },
            "reasoning": "藏族元素融合后摇属于世界音乐交叉领域，本地库覆盖薄弱需web_search。壮阔画面→梦幻/开车场景。"
        }
    },
    {
        "input": "给我整个适合在大学寝室里四个人一起打通宵游戏时放的歌单，要能提神但不能太吵被宿管投诉",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "游戏/通宵",
                "mood": "激昂"
            },
            "context": "四维+矛盾约束：场景(寝室通宵游戏)+人数(四人)+情绪(提神)+约束(音量不能太大)",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "electronic",
                "graph_scenario_filter": "工作",
                "graph_mood_filter": "激昂",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "通宵游戏需要提神(激昂/electronic)但音量受限(不能太吵)=矛盾约束。向量做动态范围控制选中等音量high歌。"
        }
    },
    {
        "input": "找几首适合拍大理洱海日落延时摄影当配乐的、空灵飘渺的女声世界音乐",
        "output": {
            "intent_type": "search",
            "parameters": {
                "genre": "世界音乐",
                "mood": "梦幻"
            },
            "context": "五维：用途(延时摄影配乐)+地点(大理洱海)+时间(日落)+风格(空灵飘渺)+类型(女声世界音乐)",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "folk",
                "graph_scenario_filter": None,
                "graph_mood_filter": "梦幻",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "世界音乐 空灵 女声 延时摄影配乐 日落 大理 洱海 etheral world music"
            },
            "reasoning": "世界音乐+空灵女声+延时摄影配乐，极小众需求需要web_search。"
        }
    },
    {
        "input": "有没有适合在下雨天的图书馆靠窗位写小说时听的、不带歌词的ambient钢琴加一点雨声采样",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "写作/创作",
                "genre": "环境音乐",
                "mood": "平静"
            },
            "context": "六维：天气(下雨)+场景(图书馆靠窗)+活动(写小说)+约束(无歌词)+流派(ambient钢琴)+音效(雨声采样)",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "classical",
                "graph_scenario_filter": "学习",
                "graph_mood_filter": "平静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "极致场景化。无歌词+ambient钢琴+雨声采样→classical+学习+平静。向量负责甄别含自然音效的纯音乐。"
        }
    },
    {
        "input": "我要一首同时致敬了Beatles和Pink Floyd的、现代制作但复古灵魂的英伦迷幻摇滚",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "Beatles, Pink Floyd",
                "genre": "迷幻摇滚",
                "region": "英国"
            },
            "context": "双实体致敬(Beatles+Pink Floyd)+风格矛盾(现代制作+复古灵魂)+流派(英伦迷幻摇滚)",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "Beatles",
                    "Pink Floyd",
                    "披头士"
                ],
                "graph_genre_filter": "rock",
                "graph_scenario_filter": None,
                "graph_mood_filter": "梦幻",
                "graph_language_filter": "English",
                "graph_region_filter": "Western",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "Beatles Pink Floyd tribute modern psychedelic rock 英伦 迷幻摇滚 现代致敬"
            },
            "reasoning": "双经典乐队致敬+新旧交叉风格，需要web_search找到带有明确致敬色彩的现代作品。"
        }
    },
    {
        "input": "推荐几首歌词是描写北京胡同生活的、用京腔唱的、带着点老北京味儿的说唱或者民谣",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "说唱/民谣",
                "language": "中文/京腔",
                "region": "北京"
            },
            "context": "五维地域文化：歌词主题(胡同生活)+方言(京腔)+流派(说唱/民谣)+地域(老北京)+文化(胡同记忆)",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "hip-hop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "怀旧",
                "graph_language_filter": "Chinese",
                "graph_region_filter": "Mainland China",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "北京胡同 京腔说唱 老北京 民谣 京味 rapper 方言说唱"
            },
            "reasoning": "北京胡同+京腔=极强地域文化标签。方言说唱/民谣覆盖度低需web_search辅助发现。"
        }
    },
    {
        "input": "给我推一首既适合弹幕空耳视频的洗脑神曲、又真的很好听不是纯粹恶搞的、最好还是日文的",
        "output": {
            "intent_type": "search",
            "parameters": {
                "genre": "流行",
                "language": "日语",
                "mood": "开心"
            },
            "context": "四维网络文化：功能(空耳弹幕视频)+约束(洗脑但好听不恶搞)+语种(日文)+网络亚文化属性",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "开心",
                "graph_language_filter": "Japanese",
                "graph_region_filter": "Japan",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "日文 空耳 洗脑 弹幕 好听 日语神曲 Japanese catchy song meme"
            },
            "reasoning": "空耳弹幕文化是网络亚文化需求，+洗脑但好听的矛盾约束+日文。需web_search深入ACG/弹幕网站。"
        }
    },
    {
        "input": "啊不对不是这首！是另一首，就是那个副歌唱了好多遍baby baby的",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "副歌含baby baby的歌曲"
            },
            "context": "否定当前播放+通过歌词片段(baby baby)从记忆中定位目标歌曲",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "歌词 baby baby 副歌"
            },
            "reasoning": "否定当前播放（不是这首）+歌词片段指代，需要web_search做歌词内容匹配。"
        }
    },
    {
        "input": "这首的调太高了我跟不上，有没有低八度的版本或者类似但调低一点的歌",
        "output": {
            "intent_type": "search",
            "parameters": {
                "song_name": "当前播放曲目"
            },
            "context": "因音域过高而要求寻找降调版本或音域更低的替代歌曲",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": ""
            },
            "reasoning": "上下文指代(这首)+音域否定(太高)+替代需求。降调版本需web_search，同风格低音域歌由向量匹配。"
        }
    },
    {
        "input": "这首歌我记得以前在某个电影里听过，是哪个电影来着",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "对当前播放歌曲曾在哪部电影中使用过的记忆模糊追问",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "上下文指代(这首歌)+影视关联查询，属于大模型知识范畴的百科问答。"
        }
    },
    {
        "input": "诶不对你搞错了，我说的不是张学友我说的是张信哲",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "张信哲"
            },
            "context": "纠正系统在上轮对话中错误解析的歌手实体",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "张信哲",
                    "Jeff Chang"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "实体纠错（张学友→张信哲），用户主动修正系统的错误解析，需要重新执行搜索。"
        }
    },
    {
        "input": "暂停一下，我接个电话回来再继续听",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "用户暂时离开请求暂停播放的控制指令",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "纯播放控制(暂停)+社交原因说明，不涉及任何推荐或搜索逻辑。"
        }
    },
    {
        "input": "不喜欢不喜欢不喜欢，下一首下一首下一首",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "放松"
            },
            "context": "连续三重否定+连续三重切歌指令表达极度不耐烦",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "极度不耐烦的重复否定表达。三重'不喜欢'+三重'下一首'=快速跳过当前并重新推荐。无约束兜底放松。"
        }
    },
    {
        "input": "我都跟你说了不要古风了怎么又给我推了一首古风",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "放松"
            },
            "context": "因系统重复违反用户之前设定的否定约束而产生的投诉和重推需求",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "多轮跨回合的否定记忆：用户之前说过不要古风，系统仍推了古风触发用户不满。需要加强记忆并重推。"
        }
    },
    {
        "input": "你推荐的歌跟我之前自己找的那些重复好多，有没有我从来没有接触过的全新领域",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "对系统推荐与已知曲库高度重叠的不满意反馈，要求突破舒适区探索全新领域",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "否定了重复推荐+要求全新领域，需要从用户画像中识别其从未涉足的流派/地域方向做破圈推荐。"
        }
    },
    {
        "input": "这些歌都太小众了我都没听过一首，能不能推点大众一些的",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "开心"
            },
            "context": "上轮推荐过于小众导致用户无法产生共鸣，要求转向热门大众歌曲",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "开心",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "否定了过度小众推荐（'都没听过'），正向要求'大众一些'→pop+开心，向量偏向高播放量热门曲。"
        }
    },
    {
        "input": "我不排斥其他语言的歌但你推荐的时候能不能把中文歌占比放到70%以上",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "放松",
                "language": "中文为主"
            },
            "context": "并非完全排除外语但设定中文歌曲比例下限的精细化偏好表达",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": "Chinese",
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "精细比例控制（中文≥70%），不是完全否定外语而是设定配额，系统需要按比例混合推荐结果。"
        }
    },
    {
        "input": "我想听啲好groovy嘅funk，嗰種James Brown嗰個era嘅old school味道",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "James Brown",
                "genre": "放克",
                "mood": "激情"
            },
            "context": "粤英混搭以James Brown为标杆的old school放克推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "James Brown",
                    "詹姆斯·布朗"
                ],
                "graph_genre_filter": "funk",
                "graph_scenario_filter": None,
                "graph_mood_filter": "激情",
                "graph_language_filter": "English",
                "graph_region_filter": "Western",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "粤英混搭（'啲好groovy嘅'=很有律动感的、'嗰種'=那种、'嗰個era嘅'=那个年代的）。James Brown锚点+funk+激情。"
        }
    },
    {
        "input": "help me搵首好chill嘅bossa nova，我想pretend自己喺巴西沙滩度假",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "波萨诺瓦",
                "mood": "放松",
                "region": "巴西"
            },
            "context": "粤英中三语极度混搭的巴西波萨诺瓦放松度假音乐请求",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "jazz",
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": "Others",
                "graph_region_filter": "Western",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "三语混搭（'help me搵首'英粤混、'好chill嘅'粤英混、'我想pretend自己喺'中英粤混）。bossa nova归入jazz+放松。"
        }
    },
    {
        "input": "有冇啲本地广州band嘅live recording，要raw同unplugged嗰種",
        "output": {
            "intent_type": "search",
            "parameters": {
                "genre": "独立/现场",
                "language": "粤语",
                "region": "广州"
            },
            "context": "粤英混搭搜索广州本地乐队的原始不插电现场录音",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "indie",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": "Cantonese",
                "graph_region_filter": "Hong Kong",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "广州 本地乐队 live recording 现场录音 unplugged 不插电 独立音乐"
            },
            "reasoning": "粤英混搭（'有冇啲'、'raw同unplugged嗰種'）。广州本地独立乐队现场录音极小众需web_search。"
        }
    },
    {
        "input": "我钟意嗰種有少少jazz元素嘅Cantopop，唔好太experimental嘅",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "粤语流行/爵士",
                "language": "粤语",
                "mood": "放松"
            },
            "context": "粤语表达融合少量爵士元素但不过于实验性的粤语流行需求",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": "Cantonese",
                "graph_region_filter": "Hong Kong",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "粤英混搭（'我钟意'=我喜欢、'有少少'=有一点、'唔好太'=不要太）。Cantopop+jazz元素+不要太experimental=pop+放松。"
        }
    },
    {
        "input": "drop一首fire嘅Mandarin rap，要嗰種flex到爆嘅gangsta style",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "说唱",
                "language": "中文",
                "mood": "激昂"
            },
            "context": "全英说唱俚语+粤语的中文gangsta说唱推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "hip-hop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "激昂",
                "graph_language_filter": "Chinese",
                "graph_region_filter": "Mainland China",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "英文说唱俚语（'drop'、'fire'、'flex'、'gangsta style'）+粤语（'嗰種'）+Mandarin rap。hip-hop+Chinese+激昂。"
        }
    },
    {
        "input": "想要啲好似Coldplay嘅Viva la Vida嗰種epic feel嘅，但系中文版嘅有冇",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "Coldplay",
                "song_name": "Viva la Vida",
                "language": "中文"
            },
            "context": "粤英混搭以Coldplay特定作品为参考寻找中文版或同类史诗感华语歌",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "Coldplay",
                    "酷玩乐队"
                ],
                "graph_genre_filter": "rock",
                "graph_scenario_filter": None,
                "graph_mood_filter": "激昂",
                "graph_language_filter": "Chinese",
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "Coldplay Viva la Vida 中文翻唱 华语 epic 史诗感 类似"
            },
            "reasoning": "粤英混搭（'好似'=好像、'嗰種'=那种、'但系'=但是、'有冇'=有没有）。特定歌曲翻唱/替代搜索需web_search。"
        }
    },
    {
        "input": "嚟首beyond嘅《真的爱你》，我想send畀我妈妈做母亲节礼物",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "真的爱你",
                "artist_name": "Beyond"
            },
            "context": "粤英混搭指定播放Beyond的母亲节经典粤语歌曲",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "粤英混搭（'嚟首'=来首、'send畀'=发给），歌手+歌名完全明确=精确点播。"
        }
    },
    {
        "input": "recommend me嗰啲好似Billie Eilish嗰種whispery vocal style嘅歌，Mandarin或English都ok",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "Billie Eilish",
                "mood": "梦幻"
            },
            "context": "粤英混搭以Billie Eilish的气声唱法为参考的不限语种推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "Billie Eilish",
                    "碧梨"
                ],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "梦幻",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "粤英混搭（'recommend me嗰啲好似'=推荐那些类似的、'嗰種whispery vocal style'=那种气声风格）。Billie Eilish+梦幻+不限语种。"
        }
    },
    {
        "input": "我想聽一啲好有summer vibe嘅Cantonese indie folk，帶住啲海邊嘅感覺",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "独立民谣",
                "language": "粤语",
                "mood": "放松"
            },
            "context": "粤英混搭的充满夏日海边氛围的粤语独立民谣推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "folk",
                "graph_scenario_filter": "散步",
                "graph_mood_filter": "放松",
                "graph_language_filter": "Cantonese",
                "graph_region_filter": "Hong Kong",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "粤英混搭（'好有summer vibe'、'帶住啲海邊嘅感覺'=带着些海边的感觉）。粤语indie folk+放松+散步。"
        }
    },
    {
        "input": "搵首Taylor Swift嘅《Love Story》嘅粤语cover version嚟听下",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "Taylor Swift",
                "song_name": "Love Story",
                "language": "粤语"
            },
            "context": "纯粤语搜索Taylor Swift特定歌曲的粤语翻唱版本",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "Taylor Swift",
                    "泰勒·斯威夫特"
                ],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": "Cantonese",
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "Taylor Swift Love Story 粤语翻唱 Cantonese cover version 广东话"
            },
            "reasoning": "粤语+英文混搭（'搵首'=找首、'嚟听下'=来听下）。跨语言翻唱需web_search搜索。"
        }
    },
    {
        "input": "你有没有发现我最近听的歌越来越丧了",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "用户的自我反思性提问，对自身近期听歌偏好变化趋势的感慨",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "自我反思式提问不需要推荐或搜索，属于大模型做情感陪伴和数据洞察分享的闲聊场景。"
        }
    },
    {
        "input": "音乐到底为什么能影响人的情绪啊这也太神奇了吧",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "对音乐心理学和人体情绪机制的好奇探讨",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "纯知识性好奇提问（音乐心理学），与具体的歌曲推荐无关，交给大模型科普。"
        }
    },
    {
        "input": "你觉得AI写的歌未来会超过人类写的歌吗",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "对AI音乐创作与人类音乐创作的哲学探讨",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "AI与人类创造力的哲学观点讨论，完全是闲聊范畴。"
        }
    },
    {
        "input": "谢谢你今天推荐的歌都很合我口味",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "用户对本次会话推荐效果的正面总结性反馈",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "正面感谢反馈，不需要执行任何动作，回复一句暖心的话即可。"
        }
    },
    {
        "input": "我在想是不是每个人对'好听'的定义都不一样",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "关于音乐审美主观性的哲学思考",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "音乐审美的主观性讨论，属于纯闲聊。"
        }
    },
    {
        "input": "你推那么多歌你自己最喜欢哪一首哈哈",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "以拟人化方式询问AI自身的音乐偏好",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "拟人化提问（AI有没有喜欢的歌），纯趣味闲聊。"
        }
    },
    {
        "input": "听了这么多歌我肚子饿了你能不能推荐个外卖",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "跳出音乐领域的跨模态请求（点外卖），超出系统能力范围",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "完全超出音乐助手能力范围的跨域请求，当闲聊处理并礼貌告知自己只能推荐音乐。"
        }
    },
    {
        "input": "不行我得走了赶地铁，帮我把今天听到的好歌全存了明天继续",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "今日播放中收藏的所有曲目"
                ]
            },
            "context": "用户紧急离开前的批量保存指令+明确的下次续听意图",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "紧急离开场景的批量保存操作（今天全部好歌），属于acquire_music的session级资产归档。"
        }
    },
    {
        "input": "不要推了不要推了今天就到这儿吧眼睛已经睁不开了",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "用户因极度疲倦主动结束本次音乐对话会话",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "会话终止信号（'不要推了'×2+'到这儿吧'），结合生理状态描述（眼睛睁不开），回一句晚安即可。"
        }
    },
    {
        "input": "我发现你跟我以前用的那个音乐app完全不一样，你真的有在理解我",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "用户对系统推荐能力的深度正向评价和情感共鸣表达",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "高度正面的产品体验反馈+情感认同，纯闲聊回应即可。"
        }
    },
    {
        "input": "我不听流行不听民谣不听摇滚不听电子不听说唱不听爵士，你还能推什么",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "古典/世界音乐",
                "mood": "平静"
            },
            "context": "连续六重流派否定后的极限挑战式测试推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "classical",
                "graph_scenario_filter": None,
                "graph_mood_filter": "平静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "六重流派否定（排除pop/folk/rock/electronic/hip-hop/jazz），剩余可选择的大类只有classical和世界音乐。极限边界测试。"
        }
    },
    {
        "input": "你推的歌单量不要超过五首，我选择恐惧症很严重",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "放松"
            },
            "context": "用户设定推荐数量上限（≤5首）以缓解选择困难症",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "数量约束（≤5首）+ 心理状态说明（选择恐惧症），核心仍是一个无具体方向的泛推荐，兜底放松。"
        }
    },
    {
        "input": "我只想听活着的歌手唱的歌，不要给我推已经去世的歌手",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "放松"
            },
            "context": "基于歌手存活状态的罕见否定约束",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "极其罕见的元数据否定约束（歌手是否在世），需要图谱或web做歌手状态验证后过滤。"
        }
    },
    {
        "input": "明天是周一不想这么快结束周末，再给我放一首歌让我多躺一会儿",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "放松"
            },
            "context": "周日晚间因不想面对周一工作日而延长慵懒时光的推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "典型的'Sunday Scaries'（周日焦虑），要求'多躺一会儿'→放松/助眠型推荐，用一首歌延长周末感。"
        }
    },
    {
        "input": "好了好了就这样吧，今天辛苦你了小助手，下次见！",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "用户以友好告别方式结束本次音乐对话",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "会话结束语+拟人化感谢（'辛苦你了小助手'），回复告别即可。"
        }
    }
,
    {
        "input": "帮我放一下朴树的《那些花儿》",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "那些花儿",
                "artist_name": "朴树"
            },
            "context": "精确点播朴树的经典校园民谣",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "歌手+歌名完全明确，直接精确点播。"
        }
    },
    {
        "input": "我想听薛之谦那首《演员》完整版的",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "演员",
                "artist_name": "薛之谦"
            },
            "context": "指定播放薛之谦热门单曲完整版本",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "精确指定歌手和歌名，且补充了'完整版'的版本偏好。"
        }
    },
    {
        "input": "放一首《夜曲》",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "夜曲"
            },
            "context": "只给出歌名的精确点播，歌手需系统推断（大概率周杰伦）",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "只有歌名没有歌手，但《夜曲》辨识度极高，系统应能自动关联周杰伦。"
        }
    },
    {
        "input": "来一首李荣浩的新歌，就是最近刚发的那首",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "李荣浩最新发行单曲",
                "artist_name": "李荣浩"
            },
            "context": "指定歌手+时间条件（最新发行）的点播",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "李荣浩 最新单曲 新歌 2024 2025"
            },
            "reasoning": "'最近刚发的'带时效性，需web_search确认最新歌曲名称后再播放。"
        }
    },
    {
        "input": "我要听黄家驹唱的那版《光辉岁月》，不要其他人翻唱的",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "光辉岁月",
                "artist_name": "黄家驹"
            },
            "context": "指定原唱歌手播放且明确排斥翻唱版本",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "歌手+歌名明确+否定翻唱，精确点播原版。"
        }
    },
    {
        "input": "播放赵雷在《歌手》节目上唱《成都》的那个现场版",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "成都",
                "artist_name": "赵雷"
            },
            "context": "指定歌手+歌名+来源节目+版本类型（综艺现场版）",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "赵雷 成都 歌手 现场版 live"
            },
            "reasoning": "指定了特定来源（歌手节目）的现场版，需web_search精确定位该版本。"
        }
    },
    {
        "input": "给我放那首'确认过眼神我遇上对的人'",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "确认过眼神"
            },
            "context": "通过歌词片段来点播歌曲",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "确认过眼神 我遇上对的人 歌曲 歌名 歌手"
            },
            "reasoning": "用户通过歌词片段来标识歌曲，需要web_search做歌词→歌名匹配。"
        }
    },
    {
        "input": "帮我搜一下陶喆的《普通朋友》然后直接放",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "普通朋友",
                "artist_name": "陶喆"
            },
            "context": "搜索+播放二合一的请求，核心意图是播放",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "虽然说了'搜一下'但紧跟'然后直接放'，核心意图是点播而非信息搜索。"
        }
    },
    {
        "input": "五月天《倔强》，谢谢",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "倔强",
                "artist_name": "五月天"
            },
            "context": "极简的歌手+歌名点播",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "极简表达直接给出了歌手和歌名，精确点播。"
        }
    },
    {
        "input": "我要听那首抖音上特别火的'你笑起来真好看'",
        "output": {
            "intent_type": "play_specific_song_online",
            "parameters": {
                "song_name": "你笑起来真好看"
            },
            "context": "通过社交平台热度+歌名进行点播",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "虽然提到了抖音但给出了完整歌名，核心是精确点播。"
        }
    },
    {
        "input": "我最近疯狂迷上了毛不易，他还有什么好听的歌推荐一下",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "毛不易"
            },
            "context": "基于当前强烈喜爱的歌手发起的作品探索推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "毛不易"
                ],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": "Chinese",
                "graph_region_filter": "Mainland China",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "明确的歌手锚点+探索性推荐需求。图谱拉取毛不易全部关联歌曲。"
        }
    },
    {
        "input": "有没有像赵雷那种感觉的民谣歌手，风格接近但不是他的",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "赵雷",
                "genre": "民谣"
            },
            "context": "以赵雷为参考锚点寻找风格相似但不同的民谣歌手",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "赵雷"
                ],
                "graph_genre_filter": "folk",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": "Chinese",
                "graph_region_filter": "Mainland China",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "赵雷作为正面锚点+限定民谣流派+排除赵雷本人=相似歌手发现。"
        }
    },
    {
        "input": "窦靖童的歌我都听完了还有谁跟她画风像的",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "窦靖童"
            },
            "context": "因穷尽某歌手全部作品后寻找风格相似的替代歌手",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "窦靖童",
                    "Leah Dou"
                ],
                "graph_genre_filter": "indie",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "已经听完窦靖童全部作品，需要找画风相近的其他歌手。"
        }
    },
    {
        "input": "推荐一些华晨宇早期还没那么疯的时候的歌",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "华晨宇",
                "mood": "温和"
            },
            "context": "限定歌手+时期（早期）+风格约束（不要太激烈）的精细推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "华晨宇",
                    "Hua Chenyu"
                ],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "治愈",
                "graph_language_filter": "Chinese",
                "graph_region_filter": "Mainland China",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "特定歌手的时期限定推荐（早期温和风格），需要图谱+向量联合筛选。"
        }
    },
    {
        "input": "我女朋友喜欢IU，有没有什么适合送给她当生日歌单的IU的歌",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "IU",
                "mood": "浪漫/温馨"
            },
            "context": "为他人（女朋友生日）基于指定歌手进行礼物型歌单推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "IU",
                    "李知恩"
                ],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "浪漫",
                "graph_language_filter": "Korean",
                "graph_region_filter": "Korea",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "以IU为歌手锚点+生日礼物场景+浪漫温馨情绪的组合型歌手推荐。"
        }
    },
    {
        "input": "邓紫棋除了那几首大众耳熟能详的以外有没有一些比较冷门的好歌",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "邓紫棋"
            },
            "context": "排除热门单曲后的冷门佳作发掘推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "邓紫棋",
                    "G.E.M."
                ],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": "Chinese",
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "邓紫棋热门歌排除后的长尾探索，图谱需要过滤掉高播放量的已知热门。"
        }
    },
    {
        "input": "李宗盛和罗大佑谁写的情歌更好听，各推荐三首让我比比",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "李宗盛, 罗大佑",
                "genre": "情歌"
            },
            "context": "双歌手对比推荐各三首情歌进行品质比较",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "李宗盛",
                    "Jonathan Lee",
                    "罗大佑",
                    "Lo Ta-yu"
                ],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "浪漫",
                "graph_language_filter": "Chinese",
                "graph_region_filter": "Taiwan",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "双歌手对比+各推三首+限定情歌，图谱需分别检索两位歌手的浪漫作品。"
        }
    },
    {
        "input": "我只知道Adele唱过Rolling in the Deep和Hello，她还有什么代表作",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "Adele"
            },
            "context": "以有限已知作品为基础扩展该歌手的代表作品认知",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "Adele",
                    "阿黛尔"
                ],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": "English",
                "graph_region_filter": "Western",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "用户已知部分作品，要求扩展到更多代表作。在图谱中排除已知两首后推荐。"
        }
    },
    {
        "input": "许嵩的歌是不是都是他自己写的？推荐几首他作词作曲特别好的",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "许嵩"
            },
            "context": "带有百科性好奇的歌手作品推荐，侧重于词曲创作质量",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "许嵩",
                    "Vae"
                ],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": "Chinese",
                "graph_region_filter": "Mainland China",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "前半句是闲聊但后半句明确要求推荐，核心还是歌手推荐。"
        }
    },
    {
        "input": "朋友推荐了一个叫落日飞车的乐队说特别好听帮我放几首试试",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "落日飞车"
            },
            "context": "来自社交推荐的新歌手试听请求",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "落日飞车",
                    "Sunset Rollercoaster"
                ],
                "graph_genre_filter": "indie",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": "Taiwan",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "朋友推荐的新乐队试听，是歌手/乐队维度的探索推荐。"
        }
    },
    {
        "input": "我是不是最近听伤感的歌太多了帮我看看我这个月的听歌情绪分布",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "请求系统分析近一个月内的情绪标签分布做个人听歌画像诊断",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "个人情绪分布分析是纯用户数据驱动的洞察需求，只能从播放记录中统计。"
        }
    },
    {
        "input": "把我收藏最多的那个流派的歌再给我多推几首",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "基于用户收藏数据中最高频流派做同类深度推荐",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'收藏最多的流派'需要从用户个人数据中统计后做扩展推荐。"
        }
    },
    {
        "input": "我平时一般都听什么类型的歌啊帮我总结一下",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "请求系统生成用户个人音乐品味画像总结",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "音乐品味总结完全依赖个人历史数据分析，不涉及任何外部推荐。"
        }
    },
    {
        "input": "根据我过去一年的听歌数据帮我生成一个年度听歌报告",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "请求生成类似Spotify Wrapped的个人年度听歌数据报告",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "年度听歌报告（类似Spotify Wrapped）是典型的用户数据聚合分析。"
        }
    },
    {
        "input": "你觉得以我的口味我会喜欢最近新出的那些歌吗帮我筛一筛",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "用个人画像作为筛选器过滤最近新发行的歌曲",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "用户画像+新歌过滤=个人偏好驱动的新歌发现，核心依赖favorites画像。"
        }
    },
    {
        "input": "挑一首你认为最能代表我音乐口味的歌",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "要求系统从用户画像中提炼出最具代表性的一首'灵魂之歌'",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'最能代表我口味'需要对用户画像做深度综合分析后选出一首极致匹配歌曲。"
        }
    },
    {
        "input": "我上周跟你聊天时你推荐的那批歌里有几首我反复听了好多遍，照那个路子再来",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "以历史推荐中被高频复听的歌曲为锚继续延伸推荐",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'反复听了好多遍'=个人交互数据中的高频复听信号，以此为锚做偏好强化推荐。"
        }
    },
    {
        "input": "我之前不是说过我不喜欢电子乐吗怎么还给我推",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "用户提醒系统应记住之前表达过的长期偏好排除",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "跨会话的偏好记忆提醒（之前说过不喜欢电子乐），需要调用长期用户偏好存储。"
        }
    },
    {
        "input": "能不能帮我把我所有的歌单按照流派重新分类整理一下",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "全部用户歌单"
                ]
            },
            "context": "请求系统对所有歌单做流派维度的自动化重新分类整理",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "全量歌单的智能分类整理是高级资产管理操作。"
        }
    },
    {
        "input": "把刚才那个歌单里重复的歌删掉，我看好像有好几首重复了",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "当前歌单"
                ]
            },
            "context": "去重清理指定歌单中的重复曲目",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "歌单去重是资产维护操作，需遍历歌单内容检测重复项。"
        }
    },
    {
        "input": "帮我把这首歌的封面图保存一下当壁纸",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "当前播放曲目"
                ]
            },
            "context": "保存当前播放歌曲的专辑封面图像用于手机壁纸",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "保存专辑封面图是跨媒体类型的资产导出操作。"
        }
    },
    {
        "input": "我刚才不小心删掉了一首歌能帮我恢复吗",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "最近删除的曲目"
                ]
            },
            "context": "撤销误操作恢复刚才删除的曲目",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "误删恢复是资产管理中的撤销操作。"
        }
    },
    {
        "input": "把《晴天》《七里香》《稻香》《青花瓷》打包成一个周杰伦精选歌单",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "周杰伦 晴天",
                    "周杰伦 七里香",
                    "周杰伦 稻香",
                    "周杰伦 青花瓷"
                ]
            },
            "context": "将指定的多首歌曲打包创建为一个新的主题歌单",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "明确列出四首歌+创建主题歌单=批量资产组织操作。"
        }
    },
    {
        "input": "能不能帮我看看我的手机存了多少首离线歌曲占了多少空间",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "离线缓存全部"
                ]
            },
            "context": "查看本地离线缓存的歌曲数量和存储空间占用",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "查看离线缓存信息是资产管理的统计查询操作。"
        }
    },
    {
        "input": "这首歌的伴奏版有吗我想下载下来自己唱",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "当前播放曲目 伴奏版"
                ]
            },
            "context": "搜索并下载当前播放歌曲的伴奏/KTV版本用于个人演唱",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": ""
            },
            "reasoning": "伴奏版下载既涉及搜索（找伴奏版本）又涉及资产获取（下载），以acquire为主。"
        }
    },
    {
        "input": "嗯...让我想想今天想听什么...算了你帮我决定吧",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "放松"
            },
            "context": "犹豫不决后将选择权完全交给系统的被动推荐模式",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "用户放弃主动选择，需要系统基于默认策略推荐。无明确约束兜底放松。"
        }
    },
    {
        "input": "我刚跟男朋友吵完架心情很差，不想听情歌但也不想听太吵的",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "治愈"
            },
            "context": "因情感事件导致的低落情绪+双重否定约束（不要情歌+不要太吵）",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "治愈",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "吵架后低落情绪+不要情歌+不要太吵=双否定取治愈中间值。需要温暖但非爱情主题的中等能量音乐。"
        }
    },
    {
        "input": "我家猫刚走了...有没有什么歌能陪我度过这段时间的",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "治愈"
            },
            "context": "因宠物离世的悲伤寻求音乐陪伴的情感疗愈推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "治愈",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "失去宠物的悲伤情境下需要温暖治愈的陪伴型音乐，不是伤感而是疗愈。"
        }
    },
    {
        "input": "今天升职了！给我来点最嗨的让我庆祝一下",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "激情/开心"
            },
            "context": "因工作晋升带来的极度喜悦需要高能量庆祝音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "开心",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "升职庆祝的极度正面情绪→开心/激情，需要高能量派对型音乐。"
        }
    },
    {
        "input": "外面在打雷下暴雨，给我来点配合这个氛围的",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "梦幻/孤独"
            },
            "context": "根据当前天气环境（暴雨雷声）匹配氛围音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "孤独",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "暴雨天气→氛围配合→孤独/梦幻的深沉音乐。环境驱动的情绪推荐。"
        }
    },
    {
        "input": "周杰伦最新一张专辑叫什么名字来着",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "对特定歌手最新专辑名称的信息查询",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "纯信息查询（专辑名称），不涉及推荐或播放，交给大模型知识库。"
        }
    },
    {
        "input": "Spotify和Apple Music哪个音质好一点",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "音乐平台音质对比的知识性讨论",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "音乐平台对比是通用知识讨论，与推荐系统功能无关。"
        }
    },
    {
        "input": "为什么有些歌在KTV里唱出来跟原唱差别那么大",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "关于KTV演唱与原唱差异原因的音乐知识探讨",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "KTV唱歌的音乐知识讨论，纯闲聊。"
        }
    },
    {
        "input": "嘿siri帮我播放...哦不对你不是siri哈哈哈",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "用户习惯性唤醒语误用后的自嘲",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "误将系统当作Siri的搞笑插曲，不包含完整的音乐请求，当闲聊处理。"
        }
    },
    {
        "input": "下午三点的办公室困得要死，给我整点能醒脑的摇滚或电子",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "工作",
                "genre": "摇滚/电子",
                "mood": "激昂"
            },
            "context": "下午工作犯困场景下的提神醒脑音乐推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "rock",
                "graph_scenario_filter": "工作",
                "graph_mood_filter": "激昂",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "下午办公室犯困→工作场景+提神→激昂+摇滚/电子。典型的场景驱动推荐。"
        }
    },
    {
        "input": "帮我搞一个做瑜伽的时候听的歌单，要那种能听到鸟叫和流水声的",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "瑜伽/冥想",
                "genre": "自然音/纯音乐",
                "mood": "平静"
            },
            "context": "瑜伽专用歌单请求，要求包含自然环境音效",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "classical",
                "graph_scenario_filter": "睡前",
                "graph_mood_filter": "平静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "瑜伽+自然音效（鸟叫流水声）→平静/睡前+向量筛选含自然采样的音轨。"
        }
    },
    {
        "input": "今晚home party大概十个人来，帮我选个从暖场到高潮到散场的全流程歌单",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "聚会",
                "mood": "多段式"
            },
            "context": "家庭派对全流程多阶段歌单编排（暖场→高潮→散场）",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": "聚会",
                "graph_mood_filter": "开心",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "聚会全流程三阶段编排需要系统做多轮不同能量级的检索并按时间线组织。"
        }
    },
    {
        "input": "Qwen3.5模型微调用什么框架比较好",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "完全跟音乐无关的技术问题",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "完全超出音乐推荐范围的技术问题，当闲聊处理并提示自己只能推荐音乐。"
        }
    },
    {
        "input": "你猜猜我现在想听什么类型的歌",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "挑战系统对用户画像的理解能力的互动式推荐",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'你猜'是对系统用户画像建模能力的测试，需要基于已有偏好数据做最优推测推荐。"
        }
    },
    {
        "input": "来首适合发朋友圈配文'今天天气真好'的歌",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "开心"
            },
            "context": "为社交媒体配文需求匹配情绪一致的歌曲",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "开心",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "朋友圈配文'天气真好'→阳光开心情绪→匹配轻快愉悦的歌曲。"
        }
    }
,
    {
        "input": "有没有好听的蓝调口琴，就那种美国南方公路电影里会出现的",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "蓝调",
                "mood": "怀旧",
                "region": "美国南部"
            },
            "context": "带有场景画面感的蓝调口琴器乐推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "blues",
                "graph_scenario_filter": "开车",
                "graph_mood_filter": "怀旧",
                "graph_language_filter": "English",
                "graph_region_filter": "Western",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "美国南方公路片→开车/旅途+蓝调口琴=blues+怀旧+Western。向量匹配口琴音色。"
        }
    },
    {
        "input": "推荐一些适合弹吉他跟弹的fingerstyle指弹曲目",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "指弹吉他",
                "mood": "放松"
            },
            "context": "以学习跟弹为目的的指弹吉他器乐推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "folk",
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "fingerstyle指弹是folk下的器乐子类，用途是学习跟弹而非纯粹欣赏。"
        }
    },
    {
        "input": "最近很想了解一下中东风格的音乐，有没有入门推荐",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "中东/世界音乐",
                "region": "中东"
            },
            "context": "对全新地域音乐风格的入门探索推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "folk",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": "Others",
                "graph_region_filter": "Western",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "中东音乐 入门 阿拉伯 波斯 推荐 Middle Eastern music beginner"
            },
            "reasoning": "中东音乐本地库覆盖薄弱，'入门推荐'需要web_search找到经典入门作品列表。"
        }
    },
    {
        "input": "介绍一些八九十年代香港电影配乐大师的作品",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "电影配乐",
                "region": "香港",
                "mood": "怀旧"
            },
            "context": "80-90年代港片配乐黄金时期的大师级作品推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "classical",
                "graph_scenario_filter": None,
                "graph_mood_filter": "怀旧",
                "graph_language_filter": None,
                "graph_region_filter": "Hong Kong",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "香港电影配乐 八九十年代 大师 黄霑 顾嘉辉 黎小田 港片原声带"
            },
            "reasoning": "港片配乐名家（黄霑/顾嘉辉等）的专业领域查询，需web_search辅助定位经典作品。"
        }
    },
    {
        "input": "给我推些真正爵士乐迷才听得懂的硬核爵士不要smooth jazz那种",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "硬核爵士/bebop",
                "mood": "激昂"
            },
            "context": "明确排斥流行化smooth jazz后转向硬核/bebop认真爵士推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "jazz",
                "graph_scenario_filter": None,
                "graph_mood_filter": "激昂",
                "graph_language_filter": "English",
                "graph_region_filter": "Western",
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'真正爵士乐迷'暗示高阶审美，排斥smooth jazz→要bebop/hard bop等学院派爵士。"
        }
    },
    {
        "input": "想听一些90后集体回忆的华语金曲，就是小时候电视上天天放的那种",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "华语流行",
                "mood": "怀旧",
                "region": "华语"
            },
            "context": "90后世代的集体怀旧华语金曲推荐（2000年代电视传播的热门歌）",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "怀旧",
                "graph_language_filter": "Chinese",
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'90后集体回忆+小时候电视'限定了约2000-2010年代的华语流行金曲，怀旧情绪驱动。"
        }
    },
    {
        "input": "有没有适合在音响上大声放、低音特别震撼的电子舞曲",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "电子舞曲/EDM",
                "mood": "激情"
            },
            "context": "以低频质感和音响体验为核心诉求的电子舞曲推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "electronic",
                "graph_scenario_filter": "聚会",
                "graph_mood_filter": "激情",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'低音震撼+音响大声放'强调低频声学体验→electronic+激情，向量筛选bass-heavy音轨。"
        }
    },
    {
        "input": "推荐一些大提琴独奏的古典小品",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "古典/大提琴",
                "mood": "平静"
            },
            "context": "限定乐器（大提琴）+演奏形式（独奏）+体裁（小品）的精确古典推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "classical",
                "graph_scenario_filter": None,
                "graph_mood_filter": "平静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "大提琴独奏古典小品→classical+平静，向量做乐器音色精确匹配。"
        }
    },
    {
        "input": "那个唱'你是我的小呀小苹果'的组合叫什么名字",
        "output": {
            "intent_type": "search",
            "parameters": {},
            "context": "通过歌词片段反向搜索歌手/组合名称",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "你是我的小呀小苹果 歌手 组合 歌名 谁唱的"
            },
            "reasoning": "歌词→歌手的反向搜索，用户不知道歌手名需要web_search做歌词匹配。"
        }
    },
    {
        "input": "我记得有一首歌的MV是在冰岛拍的特别美，但我忘了歌名了",
        "output": {
            "intent_type": "search",
            "parameters": {},
            "context": "通过MV拍摄地点的模糊记忆搜索歌曲",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "MV 冰岛拍摄 Iceland music video 歌曲"
            },
            "reasoning": "通过MV拍摄地（冰岛）这种非音乐属性来定位歌曲，只有web_search能做到。"
        }
    },
    {
        "input": "最近有没有什么音乐综艺节目值得看的",
        "output": {
            "intent_type": "search",
            "parameters": {},
            "context": "对当前热门音乐综艺节目的时效性信息搜索",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "最新 音乐综艺节目 2025 2024 推荐 好看"
            },
            "reasoning": "音乐综艺节目的时效性搜索，超出音乐推荐范畴但与音乐强相关，需web_search。"
        }
    },
    {
        "input": "帮我查一下今年格莱美最佳新人奖给了谁",
        "output": {
            "intent_type": "search",
            "parameters": {},
            "context": "特定年度音乐奖项获奖信息的精确搜索",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "格莱美 Grammy 最佳新人奖 Best New Artist 2025 获奖"
            },
            "reasoning": "年度奖项获奖信息具有强时效性，必须web_search获取。"
        }
    },
    {
        "input": "搜一下有没有关于周杰伦创作历程的纪录片",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "周杰伦"
            },
            "context": "歌手相关的影视纪录片内容搜索",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "周杰伦",
                    "Jay Chou"
                ],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "周杰伦 纪录片 创作历程 Jay Chou documentary"
            },
            "reasoning": "纪录片搜索虽然与音乐相关但属于影视内容检索，需web_search。"
        }
    },
    {
        "input": "华语乐坛有没有像Beatles那样影响了整个时代的乐队",
        "output": {
            "intent_type": "search",
            "parameters": {
                "artist_name": "Beatles",
                "region": "华语"
            },
            "context": "以Beatles为参考的华语乐坛历史影响力级别乐队搜索",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "Beatles",
                    "披头士"
                ],
                "graph_genre_filter": "rock",
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": "Chinese",
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "华语乐坛 影响力最大 乐队 Beatles 披头士 中国音乐历史 传奇乐队"
            },
            "reasoning": "跨文化对比(Beatles→华语)的音乐史级搜索，属于知识型深度查询。"
        }
    },
    {
        "input": "2024年各大平台年度最佳专辑榜单汇总",
        "output": {
            "intent_type": "search",
            "parameters": {},
            "context": "多平台年度最佳专辑榜单的聚合搜索",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "2024 年度最佳专辑 榜单 Pitchfork Apple Music Spotify 年终盘点"
            },
            "reasoning": "多平台年终榜单汇总是强时效性的聚合搜索，完全依赖web_search。"
        }
    },
    {
        "input": "最近有什么华语歌手要开演唱会的",
        "output": {
            "intent_type": "search",
            "parameters": {},
            "context": "近期华语歌手演唱会排期的时效性信息搜索",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "华语歌手 演唱会 最新 排期 2025 近期 巡演"
            },
            "reasoning": "演唱会排期是强时效性的信息搜索，需web_search获取。"
        }
    },
    {
        "input": "我正在高速上开车呢给我来点不容易犯困的",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "开车",
                "mood": "激昂"
            },
            "context": "高速公路长途驾驶防困的提神音乐推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "开车",
                "graph_mood_filter": "激昂",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "高速驾驶+防困=开车场景+激昂情绪，需要中高能量的提神型音乐。"
        }
    },
    {
        "input": "小孩要睡觉了帮我放点催眠曲",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "哄睡",
                "mood": "平静"
            },
            "context": "为幼儿入睡场景推荐的催眠曲",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "classical",
                "graph_scenario_filter": "睡前",
                "graph_mood_filter": "平静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "哄小孩睡觉→睡前/平静+classical，需要极低能量的舒缓催眠类音乐。"
        }
    },
    {
        "input": "马上要去面试了紧张得要死放点能给我壮胆的歌",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "面试准备",
                "mood": "激昂/自信"
            },
            "context": "面试前的紧张焦虑情绪下需要鼓舞士气的激励型音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "工作",
                "graph_mood_filter": "激昂",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "面试前紧张→需要激励壮胆的正能量高能歌曲。工作场景+激昂情绪。"
        }
    },
    {
        "input": "我在图书馆自习呢推荐点白噪音或者那种不影响注意力的背景音",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "学习",
                "genre": "白噪音/环境音",
                "mood": "平静"
            },
            "context": "图书馆安静环境下适合长时间专注学习的零干扰背景音",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "classical",
                "graph_scenario_filter": "学习",
                "graph_mood_filter": "平静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "图书馆自习+白噪音/背景音=学习+平静+零干扰，向量筛选极低动态范围的环境音。"
        }
    },
    {
        "input": "等下要去接我妈从机场到家大概一小时车程帮我选个歌单",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "开车",
                "mood": "放松"
            },
            "context": "接机返程途中一小时的家庭式驾车歌单",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": "开车",
                "graph_mood_filter": "放松",
                "graph_language_filter": "Chinese",
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "接妈妈的家庭驾车场景→放松/温馨+开车，考虑到有长辈在车上选择华语流行为佳。"
        }
    },
    {
        "input": "周末打算去露营有没有适合围着篝火弹吉他唱的歌",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "露营/户外",
                "genre": "民谣",
                "mood": "浪漫"
            },
            "context": "户外露营篝火场景的适合吉他弹唱的民谣歌单",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "folk",
                "graph_scenario_filter": "聚会",
                "graph_mood_filter": "浪漫",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "露营篝火弹唱→户外+民谣+浪漫，歌曲应适合原声吉他伴奏演唱。"
        }
    },
    {
        "input": "做家务的时候听什么比较带劲不觉得无聊",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "做家务",
                "mood": "开心"
            },
            "context": "家务劳动时活跃氛围缓解无聊的高能量音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": "做饭",
                "graph_mood_filter": "开心",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "做家务+带劲不无聊→开心/激昂+pop。场景近似做饭（家务类大类）。"
        }
    },
    {
        "input": "准备跑半马呢给我做个21公里的跑步歌单，从热身到冲刺节奏得递进",
        "output": {
            "intent_type": "recommend_by_activity",
            "parameters": {
                "activity": "跑步/半马",
                "mood": "多段递进"
            },
            "context": "半程马拉松全程21公里的节奏递进式跑步歌单",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "electronic",
                "graph_scenario_filter": "健身",
                "graph_mood_filter": "激昂",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "半马全程歌单需要BPM递进编排（热身低BPM→稳态中BPM→冲刺高BPM）。健身+激昂+electronic。"
        }
    },
    {
        "input": "失眠了翻来覆去睡不着来点助眠的",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "平静"
            },
            "context": "深夜失眠场景下的助眠音乐推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "classical",
                "graph_scenario_filter": "睡前",
                "graph_mood_filter": "平静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "失眠+助眠→睡前+平静+classical，需要极低刺激的舒缓音乐。"
        }
    },
    {
        "input": "考完试了终于自由了给我来点释放压力的",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "开心/释放"
            },
            "context": "考试结束后的压力释放庆祝情绪",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "开心",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'终于自由了+释放压力'是极度正面的解放感→开心/激情类高能量音乐。"
        }
    },
    {
        "input": "一个人在外地出差好孤独啊推荐点治愈的",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "治愈"
            },
            "context": "出差独处的孤独情绪下寻求温暖陪伴的治愈音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "治愈",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "出差孤独+治愈需求=温暖陪伴型音乐，不是放大孤独而是疗愈孤独。"
        }
    },
    {
        "input": "刚洗完澡躺在沙发上什么都不想干就想听着歌发呆",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "放松"
            },
            "context": "晚间洗浴后极度放松状态的无目的性音乐播放",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": "睡前",
                "graph_mood_filter": "放松",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "洗完澡+沙发发呆=极致放松状态，睡前场景+放松情绪。"
        }
    },
    {
        "input": "今天被甲方改了八遍方案我要爆炸了来点能让我平静下来的",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "平静"
            },
            "context": "工作压力爆发后急需情绪降温的镇静型音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "classical",
                "graph_scenario_filter": None,
                "graph_mood_filter": "平静",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "工作压力爆炸→急需从愤怒降温到平静，需要强力镇静效果的音乐。"
        }
    },
    {
        "input": "初恋给我发消息了心里乱得不行推荐点歌吧",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "忧郁/心乱"
            },
            "context": "初恋联系带来的情感波动下的迷茫共鸣型音乐推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "忧郁",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'初恋发消息+心里乱'是复杂情感冲击→忧郁/迷茫类情绪共鸣型音乐。"
        }
    },
    {
        "input": "下了一天的雨心情莫名其妙就低落了",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "忧郁"
            },
            "context": "因阴雨天气引发的无因式轻度情绪低落",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "忧郁",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "雨天+莫名低落=天气敏感型情绪波动→忧郁类陪伴型音乐。"
        }
    },
    {
        "input": "我现在需要一些能让我热血沸腾的音乐准备上场比赛了",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "激昂"
            },
            "context": "比赛前的肾上腺素预热需要极致热血的激励音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "rock",
                "graph_scenario_filter": "健身",
                "graph_mood_filter": "激昂",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "赛前热血沸腾→极致激昂+健身/运动场景+摇滚类高能量音乐。"
        }
    },
    {
        "input": "哈哈哈哈开玩笑的给我正经推荐",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "开心"
            },
            "context": "用户在开完玩笑后回归正轨要求正经推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "开心",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "从之前的玩笑场景中跳出，'正经推荐'但无约束。笑→开心兜底。"
        }
    },
    {
        "input": "等等让我想想...对了，有没有类似梁静茹那种温柔到骨子里的女声",
        "output": {
            "intent_type": "recommend_by_artist",
            "parameters": {
                "artist_name": "梁静茹",
                "mood": "温柔"
            },
            "context": "带有自然思考停顿的、以梁静茹为参考的温柔女声推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [
                    "梁静茹",
                    "Fish Leong"
                ],
                "graph_genre_filter": "pop",
                "graph_scenario_filter": None,
                "graph_mood_filter": "治愈",
                "graph_language_filter": "Chinese",
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "自然的思考过程表达（'等等让我想想...对了'），核心是梁静茹锚点+温柔女声推荐。"
        }
    },
    {
        "input": "话说你知道为什么很多歌听到副歌就特别上头吗",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "对副歌（hook）为什么特别抓耳的音乐理论好奇",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "音乐理论层面的好奇提问（为什么副歌抓耳），属于知识科普闲聊。"
        }
    },
    {
        "input": "我手机快没电了赶紧帮我下载两首离线歌不然一会儿没网了",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "推荐两首离线下载"
                ]
            },
            "context": "电量焦虑下的紧急离线缓存请求",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "手机没电前的紧急下载操作，核心是acquire_music的离线缓存功能。"
        }
    },
    {
        "input": "你推荐的歌能不能同时帮我在网易云和QQ音乐上都加入收藏",
        "output": {
            "intent_type": "acquire_music",
            "parameters": {
                "song_queries": [
                    "当前推荐曲目"
                ]
            },
            "context": "跨平台多端同步收藏的资产管理请求",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "跨平台同步收藏是高级资产管理操作，涉及多应用协同。"
        }
    },
    {
        "input": "给我看看我这个月一共听了多少首不同的歌",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "用户请求查看本月的独立歌曲播放数量统计",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "个人月度播放去重计数是纯用户数据统计操作。"
        }
    },
    {
        "input": "有没有什么歌是适合当手机铃声又不会太突兀被人侧目的",
        "output": {
            "intent_type": "search",
            "parameters": {},
            "context": "兼顾公共场合社交礼仪的铃声歌曲搜索",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "适合当铃声 不突兀 手机铃声 短旋律 好听 不尴尬"
            },
            "reasoning": "铃声适用性+社交考量的功能性搜索，需web_search从铃声推荐内容中匹配。"
        }
    },
    {
        "input": "你觉得听歌的时候应该用耳机还是音响好",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "关于音乐聆听设备选择的经验讨论",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "音频设备选择讨论是通用知识闲聊。"
        }
    },
    {
        "input": "我以前学过钢琴但好多年没弹了想重新捡起来有没有简单的曲子推荐",
        "output": {
            "intent_type": "recommend_by_genre",
            "parameters": {
                "genre": "古典/钢琴",
                "mood": "放松"
            },
            "context": "钢琴复习者的入门级曲目推荐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": "classical",
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": True,
                "web_search_keywords": "简单钢琴曲 入门级 钢琴初学者 推荐 重新学习 easy piano pieces"
            },
            "reasoning": "虽然涉及'学琴'但核心是推荐简单的钢琴曲目。web_search辅助找入门级曲目列表。"
        }
    },
    {
        "input": "你是真人还是AI呀",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "用户对系统身份的好奇提问",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "身份确认提问，纯闲聊。"
        }
    },
    {
        "input": "我耳机坏了只有一边有声音能不能推荐单声道也好听的歌",
        "output": {
            "intent_type": "recommend_by_mood",
            "parameters": {
                "mood": "放松"
            },
            "context": "因设备故障（单边耳机）需求适合单声道播放的音乐",
            "retrieval_plan": {
                "use_graph": True,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": "放松",
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": True,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "极其特殊的设备约束（单声道），向量需筛选非立体声依赖的歌曲。放松兜底。"
        }
    },
    {
        "input": "推荐的歌都不错但是能不能每次少推几首我有选择困难",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "对推荐数量偏好的反馈设定",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "用户偏好设定（减少推荐数量），不是具体的推荐需求而是系统行为配置。"
        }
    },
    {
        "input": "那个...嗯...就是...想听歌但不知道想听什么怎么办",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "极度纠结无法自我描述需求的用户需要系统主动画像匹配",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "'想听歌但不知道想听什么'=完全的自我表达缺失。系统只能依赖用户画像做盲推。"
        }
    },
    {
        "input": "你能不能学一下Spotify的那个Discover Weekly功能给我每周自动出个歌单",
        "output": {
            "intent_type": "recommend_by_favorites",
            "parameters": {},
            "context": "请求复刻Spotify发现周刊的个性化每周自动推荐功能",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "要求建立类似Discover Weekly的自动化推荐机制，核心依赖用户画像做周期性推荐。"
        }
    },
    {
        "input": "算了我还是自己去翻歌单了你推荐的跟我手动找的差不多",
        "output": {
            "intent_type": "general_chat",
            "parameters": {},
            "context": "用户对推荐系统差异化能力的质疑和放弃使用",
            "retrieval_plan": {
                "use_graph": False,
                "graph_entities": [],
                "graph_genre_filter": None,
                "graph_scenario_filter": None,
                "graph_mood_filter": None,
                "graph_language_filter": None,
                "graph_region_filter": None,
                "use_vector": False,
                "vector_acoustic_query": "",
                "use_web_search": False,
                "web_search_keywords": ""
            },
            "reasoning": "负面反馈+放弃使用意图，属于闲聊。系统应回应并尝试挽留。"
        }
    }
]

# Write to jsonl format
out_path = "C:\\Users\\sanyang\\sanyangworkspace\\music_recommendation\\Muisc-Research\\data\\sft\\planner_sft_data.jsonl"
with open(out_path, 'w', encoding='utf-8') as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
print(f"Generated {len(data)} items to {out_path}")
