# import deepspeed
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm


class LLMGenerateDecoder(nn.Module):
    def __init__(
            self,
            hf_model=None,
            tokenizer=None,
            cuda_device='cpu',
            loss_type='classfication',
    ):
        super().__init__()

        self.model = hf_model
        self.tokenizer = tokenizer
        self.cuda_device = cuda_device
        self.loss_type = loss_type
        self.prompt = """给定一个查询A和一个段落B，请根据段落内容判断该段落是否包含查询的答案，并给出预测结果：“是” 或 “否”。
                        查询A:{query_1}
                        段落B:{passage}
                        """
        self.yes_loc = tokenizer('是', add_special_tokens=False)['input_ids'][0]

    def forward(self, batch, labels=None):

        generated_ids_batch = self.model.generate(
            **batch,
            max_new_tokens=512,
            temperature=0.0,
            do_sample=False,
            output_logits=True,
            return_dict_in_generate=True
        )

        return generated_ids_batch

    @torch.no_grad()
    def compute_score(
            self,
            sentences_pairs,
            batch_size=12,
            max_len=512,
    ):
        '''
            sentences_pairs=[[query,title],[query1,title1],...]
        '''
        all_logits = []
        for start_index in tqdm.tqdm(range(0, len(sentences_pairs), batch_size)):
            sentences_batch = sentences_pairs[start_index:start_index + batch_size]
            batch_data = self.preprocess(sentences_batch, max_len)
            generated_ids_batch = self.forward(batch_data)
            generate_prob = torch.softmax(generated_ids_batch.logits[0], -1)
            generate_prob = generate_prob[:, self.yes_loc].view(-1, ).cpu().float().tolist()
            all_logits.extend(generate_prob)
        return all_logits

    def preprocess(self, sentences, max_len):

        message_batch = []

        for query, passage in sentences:
            prompt = self.prompt.format(query_1=query, passage=passage)
            temp_dic = {"role": "user", "content": prompt}
            message_batch.append([temp_dic])

        text_batch = self.tokenizer.apply_chat_template(
            message_batch,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs_batch = self.tokenizer(text_batch, return_tensors="pt", padding=True).to(self.model.device)

        return model_inputs_batch

    @classmethod
    def from_pretrained(
            cls,
            model_name_or_path,
            loss_type='classfication',
            num_labels=1,
            cuda_device='cpu',
    ):
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype="auto",
            device_map="balanced_low_0",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="left",
            trust_remote_code=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        reranker = cls(hf_model, tokenizer, cuda_device, loss_type)
        return reranker


def test_relecance():
    ckpt_path = "/root/autodl-tmp/miggs/llama3.1_3.18/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用cuda设备
    llmreranker = LLMGenerateDecoder.from_pretrained(ckpt_path)
    # llmreranker = nn.DataParallel(llmreranker)  # 添加并行代码  用于分布式训练
    # deepspeed.initialize(model=llmreranker)  #deepspeed 初始化
    llmreranker.eval()
    input_lst = [
        ['别克忧',
         '### 网页标题为：\n【文章】车闻|不止销量下滑，别克的这些小毛病让车主头疼!_车家号_汽车之家\n\n### 网页摘要为：\n时隔将近两年,别克凯越又宣布复产,说实话,它的回归既让人喜又让人忧,喜的是再次可以看到凯越,忧的是它能否延续曾...\n\n### 网页内容为：\n车闻|不止销量下滑，别克的这些小毛病让车主头疼！一直以来，在中国消费者心中，别克是略低于奥迪的品牌。安全、舒适、大气、美式豪华都是别克品牌在中国塑造的几个成功标签。但在近年来的发展进程中，别克突然变换定位，弃用多年积累的标识，误判中国汽车消费发展进度，忽视中国消费者审美习惯、使用习惯、用车场景需求，实行激进的年轻化战略，错判三缸发动机市场前景，过度依赖降价促销政策。很多车主同时也很无奈的表示，别克的“小毛病”就是修三大件。凯越难复活别克旗下凯越（参数|询价）这款车相信忠粉们自然也不会感到陌生，曾在国内市场上卖得异常火爆，月销量最高都达到了3万辆，但后来却选择了停产。时隔将近两年，别克凯越又宣布复产，说实话，它的回归既让人喜又让人忧，喜的是再次可以看到凯越，忧的是它能否延续曾经的辉煌。而事实证明了，车还是这个车，江湖却早已不是当年的江湖了，尽管它在售价上都体现得十分亲民，如今都跌破了7万，超高的性价比直逼国产车，8月也仅仅只卖出了2辆，这让别克感到很无奈啊。在百度搜索“中控台脱胶”或“仪表台开裂”等关键词，均出现别克凯越的身影，且占比不少。作为上汽通用旗下车型，别克凯越不到十万的售价，曾吸引了一大批注重性价比的消费者，但伴随着销量的增长，该车质量问题呈井喷式爆发。口碑的下降使别克凯越销量下滑严重，直接导致2016年8月产品停产销售。仪表台脱胶翘起，不仅会严重影响车辆内饰的美观，更会成为日常用车的潜在危险。一方面，翘起的尖锐棱角在与人体发生擦碰时可能会造成危害，对驾驶员视线也造成一定影响；另一方面，副驾驶前方开胶的面板下覆盖着安全气囊，如若发生碰撞气囊弹出，巨大的冲击力可能将仪表板一并弹起，对车内乘员造成二次伤害。别克的品控堪忧其实，导致消费者抛弃别克的另一个原因是质量堪忧，大大小小毛病集中在变速器、发动机以及各种莫名其妙、没完没了的电子件故障。很多车主无奈的表示，“别克的小毛病就是修三大件”。除此之外，别克还深陷“断轴门”的质量问题。提到断轴事件，人们尤为的重视，因为这涉及到出行的安全性，这相比发生追尾事故甚至还要严重，而对于断轴这个问题说白了就是制造厂家设计上的一大缺陷。而在高速公路上发生断轴事故更是险中险象环生。所幸这位车主并没有因为车辆断轴而遭遇严重的事故，因为当时车辆只有20码，但是如果是在高速行驶过程中车辆发生了断轴，而仅用车子走的路烂这样的借口来掩饰车辆的质量问题显然是不够的。定位尴尬的君越上汽通用显然没有扣准崛起的新晋年轻精英阶层的审美需求，对于喜欢精致生活的消费者而言，略显廉价的君越，不是他们的嗨点。新君越的用料简配非常明显，那种应有的美式豪华、大气厚重消失不见，取而代之的是廉价塑料，让消费者无法接受。再就是1.5t+7挡双离合的动力系统，也颇受诟病。（12款君越）（16款君越）别克盲目舍弃在中国积累的原始标签，导致部分喜欢别克的消费者抱憾离场，这种鲁莽，导致别克客户流失。我们来看竞品帕萨特，很长一段时间里它都带有“政府机关”用车的标签，但是帕萨特几经蜕变，却没有刻意回避这些标签，而是在此基础上扩展新的年轻消费者，并获得成功。别克独有的美式豪华、舒适、安全以及其独特的大气魅力，被最近改款的君越全部抛弃，用老百姓的审美角度去看，就是“君越不大气了！越改越像个小车！不好看，没亮点！”这样的君越显然是不足以吸引中国消费者的。-end-'],
        ['我喜欢美国', '我不喜欢美国'],
        ['泰山要多长时间爬上去',
         '爬上泰山需要1-8个小时，具体的时间需要看个人的身体素质。专业登山运动员可能只需要1个多小时就可以登顶，有些身体素质比较低的，爬的慢的就需要5个多小时了。'],
        ['鹦鹉吃自己的小鱼吗',
         '鸮鹦鹉主要是草食性,原生的植物、种子、果实及花粉等,甚至是一些树木的边材都能成为它们的食物。在1984年的一次关于鸮鹦鹉的食物及食性研究中确认了共25种的食物,并证明了它们是一种广泛的草食性生物,对于不同的乔木、灌木以及蕨类植物均感兴趣。鸮鹦鹉的喙能有效地碾磨食物,因此它们只有一个相对小的沙囊,此外,鸮鹦鹉的前肠内有细菌协助发酵及消化植物。另外它们有一套独特的习性,就是会用喙将叶片或蕨叶最具营养的部份挑选出来,难以消化的纤维部份则会留下。']
    ]

    res = llmreranker.compute_score(input_lst)

    print(res)


if __name__ == "__main__":
    test_relecance()
