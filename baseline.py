class BaselineModel:

    def __init__(self, train_df, label="OnTime_Reliability"):
        self.train_df = train_df
        self.label = label

    def predict(self, carrier, service, pod, pol):

        pred = self.train_df[
            (self.train_df["Carrier"]==carrier) & (self.train_df["Service"]==service) & \
            (self.train_df["POD"]==pod) & (self.train_df["POL"]==pol)
        ][self.label]


        return pred.iloc[0]