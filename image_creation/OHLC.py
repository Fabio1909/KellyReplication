from PIL import Image
import numpy as np

# NB. price info has to be a DataFrame with the following schema
#           ┌───────────────────────────────────────────────────────────┐
#           │                           DataFrame                       │
#           ├───────────────┬──────────┬──────────┬──────────┬──────────┤
# Index     │   Open        │   High   │   Low    │  Close   │   Vol    │
# ──────────┼───────────────┼──────────┼──────────┼──────────┼──────────┤
# DateTime  │   float64     │ float64  │ float64  │ float64  │   int64  │
#           └───────────────┴──────────┴──────────┴──────────┴──────────┘

class OHLC():
    def __init__(self, price_info, window_size, height_px, chart_px, vol_px):
        self.price_info = price_info
        self.window_size = window_size
        self.image_height = height_px
        self.chart_height_px = chart_px
        self.vol_height_px = vol_px
        self.kill = False
        self.midpoint_skipper = 0

    def step1_choose_window_to_plot(self, start, end, horizon_start, horizon_end, past_start):
        self.current_window = self.price_info.iloc[start:end].copy()
        self.horizon_start = self.price_info.iloc[horizon_start]
        self.horizon_end = self.price_info.iloc[horizon_end]
        self.m_avg_data = self.price_info.iloc[past_start:end]["Close"].copy()
        self.volume = self.current_window["Vol"].copy()
        self.current_window.drop(["Vol"], axis = 1, inplace = True)

    def calculate_moving_avg(self): 
        self.moving_average = [] 
        for i in range(0, len(self.m_avg_data) - self.window_size):
            avg_window = self.m_avg_data.iloc[i:i+self.window_size]
            avg_value = avg_window.mean()
            self.moving_average.append(avg_value)

    def step2_scale_window(self):
        _max = max(max(self.current_window.max()), max(self.moving_average))
        _min = min(min(self.current_window.min()), min(self.moving_average))
        if _max == _min:
            self.kill = True
            return
        self.current_window_scaled = self.current_window.to_numpy()
        self.current_window_scaled = np.round(((self.current_window_scaled - _min)/(_max - _min))*self.chart_height_px)
        self.current_window_scaled = self.current_window_scaled.astype(int)
        self.current_window_scaled = self.chart_height_px - self.current_window_scaled
        # moving avg
        self.moving_avg_scaled = [int(round(((item - _min)/(_max - _min))*self.chart_height_px)) for item in self.moving_average]
        self.moving_avg_scaled = [self.chart_height_px - item for item in self.moving_avg_scaled]

    def __scale_volume(self):
        _max = max(self.volume)
        _min = min(self.volume)
        if _max == _min:
            self.kill = True
            return
        self.scaled_vol = list(np.round(((self.volume - _min)/(_max - _min))*self.vol_height_px))
        self.scaled_vol = [int(item) for item in self.scaled_vol]

    def step3_create_image_object(self):
        self.width, self.height = len(self.current_window_scaled) * 3, self.image_height
        self.image = Image.new("L", (self.width, self.height), "black")
        self.pixels = self.image.load()

    def step4_draw_price_on_image(self):
        self.candle_px_counter = None
        for x in range(self.width):
            if self.candle_px_counter == None:
                self.candle_px_counter = 1 

            idx = x // 3
            if idx >= len(self.current_window_scaled):
                break  # Prevent index out of range
            
            current_candle = self.current_window_scaled[idx]
            self.__draw_price_pixels(x, current_candle)

    def __draw_price_pixels(self, x, current_candle):
        open_px = current_candle[0]
        max_px = current_candle[1]
        min_px = current_candle[2]
        close_px = current_candle[3]

        # Draw Open
        if self.candle_px_counter == 1:
            self.pixels[x, open_px] = (255)
            self.candle_px_counter += 1
        # Draw Min Max line
        elif self.candle_px_counter == 2:
            start_body = min(max_px, min_px)
            end_body = max(max_px, min_px)
            for y in range(start_body, end_body + 1):
                self.pixels[x, y] = (255)
            self.candle_px_counter += 1
        # Draw Close
        elif self.candle_px_counter == 3:
            self.pixels[x, close_px] = (255)
            self.candle_px_counter = None

    def step5_draw_volume(self):
        self.__scale_volume()
        if self.kill:
            return
        for idx, x in enumerate(range(1, self.width, 3)):
            for y in range(self.height):
                if y > self.height - self.scaled_vol[idx]:
                    self.pixels[x, y] = (80)

    def draw_moving_average(self):
        for x in range(1, self.width):
            idx = x // 3
            if self.midpoint_skipper == 0:
                current_ma_level = self.moving_avg_scaled[idx]
                self.pixels[x, current_ma_level] = (150)
                self.midpoint_skipper += 1 
            elif self.midpoint_skipper == 1:
                current_ma_level = self.moving_avg_scaled[idx]
                try:
                    next_ma_level = self.moving_avg_scaled[idx +1]
                except:
                    next_ma_level = self.moving_avg_scaled[idx]
                if current_ma_level == next_ma_level:
                    self.pixels[x, current_ma_level] = (150)
                elif current_ma_level > next_ma_level:
                    difference = current_ma_level - next_ma_level - 1 
                    if difference == 0:
                        self.pixels[x, current_ma_level] = (150)
                    elif difference == 1:
                        self.pixels[x, current_ma_level - 1] = (150)
                    elif difference % 2 == 0:
                        half_diff = int(difference / 2)
                        for y in range((next_ma_level + half_diff +1),  current_ma_level):
                            self.pixels[x,y] = (150)
                    elif difference % 2 != 0:
                        special_diff = int(difference / 2) + 1
                        for y in range(next_ma_level + special_diff, current_ma_level):
                            self.pixels[x,y] = (150)
                elif current_ma_level < next_ma_level:
                    difference = next_ma_level - current_ma_level - 1
                    if difference == 0:
                        self.pixels[x, current_ma_level] = (150)
                    elif difference == 1:
                        self.pixels[x, current_ma_level + 1] = (150)
                    elif difference % 2 == 0:
                        half_diff = int(difference / 2)
                        for y in range(current_ma_level + 1, next_ma_level - half_diff):
                            self.pixels[x,y] = (150)
                    elif difference % 2 != 0:
                        special_diff = int(difference / 2) + 1
                        for y in range(current_ma_level + 1, current_ma_level + special_diff + 1):
                            self.pixels[x,y] = (150) 
                self.midpoint_skipper += 1
            elif self.midpoint_skipper == 2:
                previous_ma_level = self.moving_avg_scaled[idx - 1]
                current_ma_level = self.moving_avg_scaled[idx]
                if current_ma_level == previous_ma_level:
                    self.pixels[x, current_ma_level] = (150)
                elif current_ma_level < previous_ma_level:
                    difference = previous_ma_level - current_ma_level - 1
                    if difference == 0:
                        self.pixels[x, current_ma_level] = (150)
                    elif difference == 1:
                        self.pixels[x, current_ma_level + 1] = (150)
                    elif difference % 2 == 0:
                        half_diff = int(difference / 2)
                        for y in range(current_ma_level + 1, previous_ma_level - half_diff):
                            self.pixels[x,y] = (150)
                    elif difference % 2 != 0:
                        special_diff = int(difference / 2)
                        for y in range(current_ma_level+1, previous_ma_level - special_diff - 1):
                            self.pixels[x,y] = (150)
                elif current_ma_level > previous_ma_level:
                    difference =  current_ma_level - previous_ma_level - 1
                    if difference == 0:
                        self.pixels[x, current_ma_level] = (150)
                    elif difference == 1:
                        self.pixels[x, current_ma_level - 1] = (150)
                    elif difference % 2 == 0:
                        half_diff = int(difference / 2)
                        for y in range(previous_ma_level + half_diff +1, current_ma_level):
                            self.pixels[x,y] = (150)
                    elif difference % 2 != 0:
                        special_diff = int(difference / 2)
                        for y in range(current_ma_level - special_diff, current_ma_level):
                            self.pixels[x,y] = (150)
                self.midpoint_skipper = 0

    def step6_annotate_image(self):
        start_price = self.horizon_start["Open"]
        end_price = self.horizon_end["Close"]
        self.label = True if end_price > start_price else False

    def step7_save_img(self, directory_not_complete):
        complete_directory = f"{directory_not_complete}_{self.label}.png"
        self.image.save(complete_directory)

    def step8_clear_window(self):
        del self.current_window_scaled
        del self.current_window
        self.midpoint_skipper = 0 

    def plot_window(self, start, end, horizon_start, horizon_end, past_start, directory_not_complete):
        self.step1_choose_window_to_plot(start, end, horizon_start, horizon_end, past_start)
        # Some checks to ensure everything works as intended
        if self.m_avg_data.empty:
            print("Skipping plot as m_avg_data is empty.")
            return  
        if self.current_window.isnull().any().any():
            print("Skipping plot as current_window contains NaN values.")
            return
        if self.horizon_start is None or self.horizon_end is None:
            print("Either horizon_start or horizon_end is None.")
            return
        if self.m_avg_data.isnull().any():
            print("Skipping plot as moving average data contains NaN values.")
            return
        if self.volume.isnull().any():
            print("Skipping plot cause volume data is null")
            return

        self.calculate_moving_avg()
        self.step2_scale_window()
        if self.kill:
            print("There is a problem with the scaling min and max are the same not able to plot")
            self.kill = False
            return
        self.step3_create_image_object()
        self.step4_draw_price_on_image()
        self.step5_draw_volume()
        if self.kill:
            print("Problem with volume")
            self.kill = False
            return
        self.draw_moving_average()
        self.midpoint_skipper = 0 
        self.step6_annotate_image()
        self.step7_save_img(directory_not_complete)
        self.step8_clear_window()


# the end
