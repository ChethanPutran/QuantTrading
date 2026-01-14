# stocks = []
# with open("todays_stocks.txt",'r') as f:
#     lines =  f.readlines()

#     for i,line in enumerate(lines):
#         if i%3 == 0:
#             stocks.append(line.strip().strip("\n"))

# print(stocks)
# with open("Test.txt",'w') as f:
#     f.writelines(stocks)



from data.data_fetcher import get_todays_nifty_data
from features.pattern_detector import Tool


status, data = get_todays_nifty_data()

tool = Tool()

data = tool.capture_momentum(data,append=True)

