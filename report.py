from reportlab.pdfbase import pdfmetrics   # 注册字体
from reportlab.pdfbase.ttfonts import TTFont # 字体类
from reportlab.platypus import Table, SimpleDocTemplate, Paragraph, Image, PageBreak, BaseDocTemplate, Spacer  # 报告内容相关类
from reportlab.lib.pagesizes import letter  # 页面的标志尺寸(8.5*inch, 11*inch)
from reportlab.lib.styles import getSampleStyleSheet  # 文本样式
from reportlab.lib import colors  # 颜色模块
from reportlab.graphics.charts.barcharts import VerticalBarChart  # 图表类
from reportlab.graphics.charts.lineplots import SimpleTimeSeriesPlot
from reportlab.graphics.charts.legends import Legend  # 图例类
from reportlab.graphics.shapes import Drawing  # 绘图工具
from reportlab.lib.units import cm  # 单位：cm

import pandas as pd
# 注册字体(提前准备好字体文件, 如果同一个文件需要多种字体可以注册多个)
# pdfmetrics.registerFont(TTFont('SimSun', 'SimSun.ttf'))
class Graphs:
    # 绘制标题
    @staticmethod
    def draw_title(title: str):
        # 获取所有样式表
        style = getSampleStyleSheet()
        # 拿到标题样式
        ct = style['Heading1']
        # 单独设置样式相关属性
        #ct.fontName = 'SimSun'      # 字体名
        ct.fontSize = 18            # 字体大小
        ct.leading = 50             # 行间距
        ct.textColor = colors.black     # 字体颜色
        ct.alignment = 1    # 居中
        ct.bold = True
        # 创建标题对应的段落，并且返回
        return Paragraph(title, ct)
      
  # 绘制小标题
    @staticmethod
    def draw_little_title(title: str):
        # 获取所有样式表
        style = getSampleStyleSheet()
        # 拿到标题样式
        ct = style['Normal']
        # 单独设置样式相关属性
        #ct.fontName = 'SimSun'  # 字体名
        ct.fontSize = 15  # 字体大小
        ct.leading = 30  # 行间距
        ct.textColor = colors.black  # 字体颜色
        # 创建标题对应的段落，并且返回
        return Paragraph(title, ct)
 
    # 绘制普通段落内容
    @staticmethod
    def draw_text(text: str):
        # 获取所有样式表
        style = getSampleStyleSheet()
        # 获取普通样式
        ct = style['Normal']
        #ct.fontName = 'SimSun'
        ct.fontSize = 12
        ct.wordWrap = 'CJK'     # 设置自动换行
        ct.alignment = 0        # 左对齐
        ct.firstLineIndent = 16     # 第一行开头空格
        ct.leading = 25 # 行间距
        return Paragraph(text, ct)
 
    # 以DataFrame繪製表格，可指定index欄的名稱
    @staticmethod
    def draw_table(df, index_name="", item_name=""):
        if type(df) == pd.Series:
            df = df.to_frame(name=item_name)
            print(df)
            print(df.columns)
        
        # 列宽度
        # col_width = 120
        style = [
            #('FONTNAME', (0, 0), (-1, -1), 'SimSun'),  # 字体
            ('FONTSIZE', (0, 0), (-1, 0), 12),  # 第一行的字体大小
            ('FONTSIZE', (0, 1), (-1, -1), 10),  # 第二行到最后一行的字体大小
            ('BACKGROUND', (0, 0), (-1, 0), '#d5dae6'),  # 设置第一行背景颜色
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # 第一行水平居中
            ('ALIGN', (0, 1), (-1, -1), 'LEFT'),  # 第二行到最后一行左右左对齐
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # 所有表格上下居中对齐
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),  # 设置表格内文字颜色
            ('GRID', (0, 0), (-1, -1), 1, colors.black),  # 设置表格框线为grey色，线宽为0.5
            # ('SPAN', (0, 1), (0, 2)),  # 合并第一列二三行
            # ('SPAN', (0, 3), (0, 4)),  # 合并第一列三四行
            # ('SPAN', (0, 5), (0, 6)),  # 合并第一列五六行
            # ('SPAN', (0, 7), (0, 8)),  # 合并第一列五六行
        ]
        # table = Table(data, colWidths=col_width, style=style)
        
        # 目前無任何樣式(Style)
        # 待改：用reset_index即可
        # 因df.values不會轉換index欄位，故複製一次插入於最前面
        header = [tuple(df.columns.insert(0, index_name))]
        df.insert(0,"",df.index)
        table = Table(header+list(df.values), style=style, hAlign='LEFT')
        return table
 
    # 创建图表
    @staticmethod
    def draw_bar(bar_data: list, ax: list, items: list):
        drawing = Drawing(500, 250)
        bc = VerticalBarChart()
        bc.x = 45       # 整个图表的x坐标
        bc.y = 45      # 整个图表的y坐标
        bc.height = 200     # 图表的高度
        bc.width = 350      # 图表的宽度
        bc.data = bar_data
        bc.strokeColor = colors.black       # 顶部和右边轴线的颜色
        #bc.valueAxis.valueMin = 5000           # 设置y坐标的最小值
        #bc.valueAxis.valueMax = 26000         # 设置y坐标的最大值
        #bc.valueAxis.valueStep = 2000         # 设置y坐标的步长
        bc.categoryAxis.labels.dx = 2
        bc.categoryAxis.labels.dy = -8
        bc.categoryAxis.labels.angle = 20
        bc.categoryAxis.categoryNames = ax
 
        # 图示
        leg = Legend()
        #leg.fontName = 'SimSun'
        leg.alignment = 'right'
        leg.boxAnchor = 'ne'
        leg.x = 475         # 图例的x坐标
        leg.y = 240
        leg.dxTextSpace = 10
        leg.columnMaximum = 3
        leg.colorNamePairs = items
        drawing.add(leg)
        drawing.add(bc)
        return drawing
 
    @staticmethod
    def draw_lineChart(bar_data: list, ax: list, items: list):
        drawing = Drawing(500, 250)
        bc = SimpleTimeSeriesPlot()
        bc.x = 45       # 整个图表的x坐标
        bc.y = 45      # 整个图表的y坐标
        bc.height = 200     # 图表的高度
        bc.width = 350      # 图表的宽度
        bc.data = bar_data
        bc.strokeColor = colors.black       # 顶部和右边轴线的颜色
        #bc.valueAxis.valueMin = 5000           # 设置y坐标的最小值
        #bc.valueAxis.valueMax = 26000         # 设置y坐标的最大值
        #bc.valueAxis.valueStep = 2000         # 设置y坐标的步长
        bc.categoryAxis.labels.dx = 2
        bc.categoryAxis.labels.dy = -8
        bc.categoryAxis.labels.angle = 20
        bc.categoryAxis.categoryNames = ax
 
        # 图示
        leg = Legend()
        leg.alignment = 'right'
        leg.boxAnchor = 'ne'
        leg.x = 475         # 图例的x坐标
        leg.y = 240
        leg.dxTextSpace = 10
        leg.columnMaximum = 3
        leg.colorNamePairs = items
        drawing.add(leg)
        drawing.add(bc)
        return drawing

    # 绘制图片
    @staticmethod
    def draw_img(path):
        img = Image(path)       # 读取指定路径下的图片
        img.drawWidth = 10*cm        # 设置图片的宽度
        img.drawHeight = 8*cm       # 设置图片的高度
        return img

if __name__ == '__main__':
    # 创建内容对应的空列表
    content = list()
    # 添加标题
    content.append(Graphs.draw_title('Title'))
    # 添加段落文字
    content.append(Graphs.draw_text('Hello World'))
    content.append(Graphs.draw_little_title('Hello World - 2'))
    # 添加表格
    data = [
        ('A', '18.5K', '25%'),
        # ('B', '25.5K', '14%'),
        # ('C', '29.3K', '10%'),
    ]
    
    data = data+list(example_df.values)
    content.append(Graphs.draw_table(data))
    # 生成图表
    content.append(Graphs.draw_title(''))
    content.append(Graphs.draw_little_title('Situation'))
    b_data = [(25400, 12900, 20100, 20300, 20300, 17400), (15800, 9700, 12982, 9283, 13900, 7623)]
    ax_data = ['BeiJing', 'ChengDu', 'ShenZhen', 'ShangHai', 'HangZhou', 'NanJing']
    leg_items = [(colors.red, 'Salary'), (colors.green, 'Volume')]
    content.append(Graphs.draw_bar(b_data, ax_data, leg_items))
    content.append(Graphs.draw_lineChart(b_data, ax_data, leg_items))
    
    # 生成pdf文件
    #doc = SimpleDocTemplate('report.pdf', pagesize=letter)
    doc = BaseDocTemplate('report.pdf', pagesize=letter)
    
    doc.build(content)