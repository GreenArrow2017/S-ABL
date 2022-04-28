import cv2
import paddlehub as hub
import ddparser
# 单条句子"original_text":"一辆压路机每分钟行驶50米，压路的宽度为3米．如果行驶压路机12分钟，可以压路多少平方米？",
sentence = "早晨 教室 里 有 36 名 学生 ， 其中 女生 占 教室 里 总 人数 的 (5/9) ， 后来 又 来 了 几名 女生 ， 这时 女生 占 教室 里 总 人数 的 (11/19) ， 后来 又 来 了 几名 女生 ？"
seg = sentence.split(' ')

import cv2
import paddlehub as hub

module = hub.Module(name="ddparser")


test_tokens = [['百度', '是', '一家', '高科技', '公司']]
results = ddparser.parse_seg([seg], return_visual = True)
print(results)

result = results[0]
data = module.visualize(result['word'],result['head'],result['deprel'])
# or data = result['visual']
cv2.imwrite('test.jpg',data)
