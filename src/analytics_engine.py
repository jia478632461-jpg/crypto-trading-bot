"""Analytics Engine - MarketRegime, TechnicalAnalyzer, MarketRegimeDetector"""
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple

class MarketRegime(Enum):
    BULL_TREND = "bull_trend"; BEAR_TREND = "bear_trend"; SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"; LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"; REVERSAL = "reversal"; UNKNOWN = "unknown"

class TechnicalAnalyzer:
    @staticmethod
    def sma(c, p): return pd.Series(c).rolling(p).mean().values
    @staticmethod
    def ema(c, p): return pd.Series(c).ewm(span=p, adjust=False).mean().values
    @staticmethod
    def adx(hi, lo, c, p=14):
        hp = pd.Series(hi).diff(); lm = -pd.Series(lo).diff()
        hp[hp < 0] = 0; lm[lm < 0] = 0
        tr = pd.concat([pd.Series(hi)-pd.Series(lo), abs(pd.Series(hi)-pd.Series(c).shift(1)), abs(pd.Series(lo)-pd.Series(c).shift(1))], axis=1).max(axis=1)
        atr = tr.rolling(p).mean()
        pdi = 100*(hp.rolling(p).mean()/atr); mdi = 100*(lm.rolling(p).mean()/atr)
        dx = 100*abs(pdi-mdi)/(pdi+mdi+1e-10)
        return dx.rolling(p).mean().values, pdi.values, mdi.values
    @staticmethod
    def supertrend(hi, lo, c, p=10, mult=3.0):
        hl2 = (hi+lo)/2; atr = TechnicalAnalyzer.atr(hi, lo, c, p)
        upper = hl2+mult*atr; lower = hl2-mult*atr
        st = np.zeros_like(c); d = np.zeros_like(c)
        for i in range(1, len(c)):
            if c[i] > upper[i-1]: d[i]=1; st[i]=lower[i]
            elif c[i] < lower[i-1]: d[i]=-1; st[i]=upper[i]
            else: d[i]=d[i-1]; st[i]=st[i-1]
        return st, d
    @staticmethod
    def aroon(hi, lo, p=25):
        au = pd.Series(hi).rolling(p+1).apply(lambda x: float(np.argmax(x))/p*100, raw=True)
        ad = pd.Series(lo).rolling(p+1).apply(lambda x: float(np.argmin(x))/p*100, raw=True)
        return au.values, ad.values, (au-ad).values
    @staticmethod
    def ichimoku(hi, lo, c):
        nh=pd.Series(hi).rolling(9).max(); nl=pd.Series(lo).rolling(9).min()
        tk=(nh+nl)/2; p26h=pd.Series(hi).rolling(26).max(); p26l=pd.Series(lo).rolling(26).min()
        kj=(p26h+p26l)/2; sa=((tk+kj)/2).shift(26)
        p52h=pd.Series(hi).rolling(52).max(); p52l=pd.Series(lo).rolling(52).min()
        sb=((p52h+p52l)/2).shift(26); ck=pd.Series(c).shift(-26)
        return {"tenkan_sen":tk.values,"kijun_sen":kj.values,"senkou_span_a":sa.values,"senkou_span_b":sb.values,"chikou_span":ck.values}
    @staticmethod
    def rsi(c, p=14):
        d=pd.Series(c).diff(); g=d.where(d>0,0.0); l=(-d).where(d<0,0.0)
        ag=g.ewm(com=p-1,min_periods=p).mean(); al=l.ewm(com=p-1,min_periods=p).mean()
        return (100-100/(1+ag/(al+1e-10))).values
    @staticmethod
    def stochastic(hi, lo, c, kp=14, dp=3):
        lm=pd.Series(lo).rolling(kp).min(); hm=pd.Series(hi).rolling(kp).max()
        k=100*(c-lm)/(hm-lm+1e-10)
        return k.values, k.rolling(dp).mean().values
    @staticmethod
    def stochastic_rsi(c, p=14, kp=3, dp=3):
        r=pd.Series(TechnicalAnalyzer.rsi(c, p))
        s=(r-r.rolling(p).min())/(r.rolling(p).max()-r.rolling(p).min()+1e-10)
        k=s.rolling(kp).mean()*100
        return k.values, k.rolling(dp).mean().values
    @staticmethod
    def cci(hi, lo, c, p=20):
        typ=pd.Series((hi+lo+c)/3); sma=typ.rolling(p).mean()
        mad=typ.rolling(p).apply(lambda x: np.abs(x-x.mean()).mean(), raw=True)
        return ((typ-sma)/(0.015*mad+1e-10)).values
    @staticmethod
    def williams_r(hi, lo, c, p=14):
        hh=pd.Series(hi).rolling(p).max(); ll=pd.Series(lo).rolling(p).min()
        return (-100*(hh-c)/(hh-ll+1e-10)).values
    @staticmethod
    def roc(c, p=12):
        cs=pd.Series(c); sh=cs.shift(p)
        return (100*(cs/sh-1)).values
    @staticmethod
    def momentum(c, p=10):
        return pd.Series(c).diff(p).values
    @staticmethod
    def macd(c, f=12, s=26, sp=9):
        ef=pd.Series(c).ewm(span=f,adjust=False).mean()
        es=pd.Series(c).ewm(span=s,adjust=False).mean()
        ml=ef-es; sl=ml.ewm(span=sp,adjust=False).mean()
        return ml.values, sl.values, (ml-sl).values
    @staticmethod
    def ao(hi, lo, f=5, s=34):
        m=pd.Series((hi+lo)/2)
        return (m.rolling(f).mean()-m.rolling(s).mean()).values
    @staticmethod
    def kst(c):
        r1=TechnicalAnalyzer.roc(c,10); r2=TechnicalAnalyzer.roc(c,15)
        r3=TechnicalAnalyzer.roc(c,20); r4=TechnicalAnalyzer.roc(c,30)
        k=pd.Series(r1).rolling(10).mean()*1+pd.Series(r2).rolling(10).mean()*2+pd.Series(r3).rolling(10).mean()*3+pd.Series(r4).rolling(10).mean()*4
        return k.values, k.rolling(9).mean().values
    @staticmethod
    def trix(c, p=15):
        e1=pd.Series(c).ewm(span=p,adjust=False).mean()
        e2=e1.ewm(span=p,adjust=False).mean(); e3=e2.ewm(span=p,adjust=False).mean()
        tx=100*e3.pct_change()
        return tx.values, tx.rolling(9).mean().values
    @staticmethod
    def dpo(c, p=20):
        return (c - pd.Series(c).rolling(p).mean().shift(p//2+1).values)
    @staticmethod
    def atr(hi, lo, c, p=14):
        tr=pd.concat([pd.Series(hi)-pd.Series(lo), abs(pd.Series(hi)-pd.Series(c).shift(1)), abs(pd.Series(lo)-pd.Series(c).shift(1))], axis=1).max(axis=1)
        return tr.rolling(p).mean().values
    @staticmethod
    def natr(hi, lo, c, p=14):
        return 100*TechnicalAnalyzer.atr(hi, lo, c, p)/(c+1e-10)
    @staticmethod
    def bollinger_bands(c, p=20, sd=2.0):
        ma=pd.Series(c).rolling(p).mean(); s=pd.Series(c).rolling(p).std()
        u=(ma+sd*s).values; l=(ma-sd*s).values; mv=ma.values; bp=(c-l)/(u-l+1e-10)
        return mv, u, l, bp
    @staticmethod
    def keltner_channel(hi, lo, c, p=20, mult=2.0):
        em=pd.Series(c).ewm(span=p,adjust=False).mean().values
        at=TechnicalAnalyzer.atr(hi, lo, c, p)
        return em, em+mult*at, em-mult*at
    @staticmethod
    def donchian_channel(hi, lo, p=20):
        u=pd.Series(hi).rolling(p).max().values; l=pd.Series(lo).rolling(p).min().values
        return u, l, (u+l)/2
    @staticmethod
    def obv(c, v):
        o=np.zeros_like(c)
        for i in range(1,len(c)):
            if c[i]>c[i-1]: o[i]=o[i-1]+v[i]
            elif c[i]<c[i-1]: o[i]=o[i-1]-v[i]
            else: o[i]=o[i-1]
        return o
    @staticmethod
    def mfi(hi, lo, c, v, p=14):
        tp=(hi+lo+c)/3; rf=tp*v; pp=np.zeros_like(c); np_=np.zeros_like(c)
        for i in range(1,len(c)):
            if tp[i]>tp[i-1]: pp[i]=rf[i]
            else: np_[i]=rf[i]
        ps=pd.Series(pp).rolling(p).sum(); ns=pd.Series(np_).rolling(p).sum()
        return (100-100/(1+ps/(ns+1e-10))).values
    @staticmethod
    def adl(hi, lo, c, v):
        mf=((c-lo)-(hi-c))/(hi-lo+1e-10)*v; a=np.zeros_like(c)
        for i in range(1,len(c)): a[i]=a[i-1]+mf[i]
        return a
    @staticmethod
    def cmf(hi, lo, c, v, p=20):
        mf=pd.Series(((c-lo)-(hi-c))/(hi-lo+1e-10)*v)
        return (mf.rolling(p).sum()/pd.Series(v).rolling(p).sum()).values
    @staticmethod
    def vpt(c, v):
        p=np.zeros_like(c)
        for i in range(1,len(c)): p[i]=p[i-1]+v[i]*(c[i]-c[i-1])/(c[i-1]+1e-10)
        return p
    @staticmethod
    def nvi(c, v):
        n=np.ones_like(c)*1000
        for i in range(1,len(c)):
            if v[i]<v[i-1]: n[i]=n[i-1]+(c[i]-c[i-1])/(c[i-1]+1e-10)
            else: n[i]=n[i-1]
        return n
    @staticmethod
    def vroc(v, p=14):
        vs=pd.Series(v); sh=vs.shift(p)
        return (100*(vs/sh-1)).values
    @staticmethod
    def pivot_points(hi, lo, c):
        p=(hi+lo+c)/3
        return {"pivot":p,"r1":2*p-hi,"r2":p+(hi-lo),"r3":hi+2*(p-lo),"s1":2*p-hi,"s2":p-(hi-lo),"s3":lo-2*(hi-p)}
    @staticmethod
    def fibonacci_retracement(hi, lo):
        h,l,d=hi[-1],lo[-1],hi[-1]-lo[-1]
        return {"level_0":h,"level_236":h-0.236*d,"level_382":h-0.382*d,"level_500":h-0.500*d,"level_618":h-0.618*d,"level_786":h-0.786*d,"level_100":l}
    @staticmethod
    def support_resistance_levels(c, w=20, thr=0.02):
        sup,res=[],[]
        for i in range(w,len(c)-w):
            win=c[i-w:i+w+1]
            if c[i]==win.min() and win.min()/win.max()<(1-thr): sup.append(c[i])
            if c[i]==win.max() and win.min()/win.max()<(1-thr): res.append(c[i])
        def cluster(ls,md=0.01):
            if not ls: return []
            ls=sorted(ls); cl,cur=[],[ls[0]]
            for lv in ls[1:]:
                if abs(lv-np.mean(cur))/(np.mean(cur)+1e-10)<md: cur.append(lv)
                else: cl.append(np.mean(cur)); cur=[lv]
            cl.append(np.mean(cur)); return cl
        return cluster(sup),cluster(res)

    def compute_all(self, df):
        c=np.asarray(df["close"],dtype=float)
        h=np.asarray(df["high"],dtype=float)
        lo=np.asarray(df["low"],dtype=float)
        v=np.asarray(df["volume"],dtype=float) if "volume" in df.columns else np.ones_like(c)
        n=len(c)
        def lv(arr,d=0.0):
            a=np.asarray(arr,dtype=float); mask=~np.isnan(a)
            return float(a[mask][-1]) if np.any(mask) else float(d)
        ma7=self.sma(c,7); ma14=self.sma(c,14); ma25=self.sma(c,25)
        ma50=self.sma(c,50); ma99=self.sma(c,99); ma200=self.sma(c,200)
        ema12=self.ema(c,12); ema26=self.ema(c,26)
        adx_a,pdi,mdi=self.adx(h,lo,c)
        st_a,st_d=self.supertrend(h,lo,c)
        ichi=self.ichimoku(h,lo,c); au,ad,ao_sc=self.aroon(h,lo)
        rsi_a=self.rsi(c); sk,sd=self.stochastic(h,lo,c); srk,srd=self.stochastic_rsi(c)
        cci_a=self.cci(h,lo,c); wr=self.williams_r(h,lo,c)
        roc_a=self.roc(c); mom_a=self.momentum(c)
        macd_l,sig_l,hist=self.macd(c); ao_a=self.ao(h,lo)
        kst_a,kst_s=self.kst(c); trix_a,trix_s=self.trix(c); dpo_a=self.dpo(c)
        atr_a=self.atr(h,lo,c); natr_a=self.natr(h,lo,c)
        bb_m,bb_u,bb_l,bb_p=self.bollinger_bands(c)
        kc_m,kc_u,kc_l=self.keltner_channel(h,lo,c)
        dc_u,dc_l,dc_mv=self.donchian_channel(h,lo)
        obv_a=self.obv(c,v); mfi_a=self.mfi(h,lo,c,v); adl_a=self.adl(h,lo,c,v)
        cmf_a=self.cmf(h,lo,c,v); vpt_a=self.vpt(c,v); nvi_a=self.nvi(c,v); vroc_a=self.vroc(v)
        pivots=self.pivot_points(h,lo,c); fib=self.fibonacci_retracement(h,lo)
        supp,ress=self.support_resistance_levels(c)
        vol5=float(c[-5:].std()/c[-5:].mean()) if n>=5 else 0.0
        vol20=float(c[-20:].std()/c[-20:].mean()) if n>=20 else 0.0
        vr_val=float(v[-1]/v[-20:].mean()) if n>=20 else 1.0
        bull_score=0.0
        if c[-1]>ma50[-1]: bull_score+=1
        if ma7[-1]>ma25[-1]: bull_score+=1
        if ema12[-1]>ema26[-1]: bull_score+=1
        if ma25[-1]>ma99[-1]: bull_score+=1
        bull_score=bull_score/4.0
        return {
            "close":lv(c),"volume":lv(v),
            "sma7":lv(ma7),"sma14":lv(ma14),"sma25":lv(ma25),"sma50":lv(ma50),"sma99":lv(ma99),"sma200":lv(ma200),
            "ema12":lv(ema12),"ema26":lv(ema26),
            "adx":lv(adx_a),"plus_di":lv(pdi),"minus_di":lv(mdi),
            "supertrend":lv(st_a),"st_direction":lv(st_d),
            "aroon_up":lv(au),"aroon_down":lv(ad),"aroon_osc":lv(ao_sc),
            "tenkan_sen":lv(ichi["tenkan_sen"]),"kijun_sen":lv(ichi["kijun_sen"]),
            "rsi":lv(rsi_a),"stoch_k":lv(sk),"stoch_d":lv(sd),
            "stoch_rsi_k":lv(srk),"stoch_rsi_d":lv(srd),
            "cci":lv(cci_a),"williams_r":lv(wr),
            "roc":lv(roc_a),"momentum":lv(mom_a),
            "macd_line":lv(macd_l),"signal_line":lv(sig_l),"macd_histogram":lv(hist),
            "ao":lv(ao_a),"kst":lv(kst_a),"kst_signal":lv(kst_s),
            "trix":lv(trix_a),"trix_signal":lv(trix_s),"dpo":lv(dpo_a),
            "atr":lv(atr_a),"natr":lv(natr_a),
            "bb_mid":lv(bb_m),"bb_upper":lv(bb_u),"bb_lower":lv(bb_l),"bb_position":lv(bb_p),
            "keltner_mid":lv(kc_m),"keltner_upper":lv(kc_u),"keltner_lower":lv(kc_l),
            "donchian_upper":lv(dc_u),"donchian_lower":lv(dc_l),
            "obv":lv(obv_a),"mfi":lv(mfi_a),"adl":lv(adl_a),
            "cmf":lv(cmf_a),"vpt":lv(vpt_a),"nvi":lv(nvi_a),"vroc":lv(vroc_a),
            "pivot":lv(pivots["pivot"]),
            "r1":lv(pivots["r1"]),"r2":lv(pivots["r2"]),"r3":lv(pivots["r3"]),
            "s1":lv(pivots["s1"]),"s2":lv(pivots["s2"]),"s3":lv(pivots["s3"]),
            "fib_236":fib["level_236"],"fib_382":fib["level_382"],"fib_500":fib["level_500"],"fib_618":fib["level_618"],"fib_786":fib["level_786"],
            "num_supports":len(supp),"num_resistances":len(ress),
            "volatility_5":vol5,"volatility_20":vol20,
            "price_change":float((c[-1]-c[-2])/c[-2]) if n>=2 else 0.0,
            "price_change_5":float((c[-1]-c[-6])/c[-6]) if n>=6 else 0.0,
            "price_change_20":float((c[-1]-c[-21])/c[-21]) if n>=21 else 0.0,
            "high_low_ratio":float(h[-1]/lo[-1]) if lo[-1]>0 else 1.0,
            "ma_bull_score":bull_score,
            "volume_ratio":vr_val,
        }

    def score_technical(self, ind):
        c=ind.get("close",1.0); ma7=ind.get("sma7",c); ma25=ind.get("sma25",c); ma50=ind.get("sma50",c)
        rsi=ind.get("rsi",50); sk=ind.get("stoch_k",50)
        adx=ind.get("adx",0)/100.0; pdi=ind.get("plus_di",0); mdi=ind.get("minus_di",0)
        bb=ind.get("bb_position",0.5); natr=ind.get("natr",0)/100.0
        vr=ind.get("volume_ratio",1.0); mfi=(ind.get("mfi",50)-50)/50.0
        cmf=ind.get("cmf",0); roc=ind.get("roc",0)/100.0
        r1=ind.get("r1",c); s1=ind.get("s1",c)
        # Trend
        t1=1.0 if c>ma7>ma25>ma50 else (-1.0 if c<ma7<ma25<ma50 else 0.0)
        t2=adx*(1.0 if pdi>mdi else -1.0)
        ma_bs=ind.get("ma_bull_score",0); aroon=ind.get("aroon_osc",0)/100.0
        trend=(t1+t2+ma_bs+aroon)/4.0
        scores_trend=max(-1.0,min(1.0,trend))
        # Momentum
        m1=1.0 if rsi<30 else (-1.0 if rsi>70 else (rsi-50)/20.0)
        m2=1.0 if sk<20 else (-1.0 if sk>80 else (sk-50)/30.0)
        macdh=ind.get("macd_histogram",0)/(abs(c)+1e-10)
        roc_v=roc; cci_v=ind.get("cci",0)/100.0
        mom_vals=[m1,m2,float(np.tanh(macdh)),float(np.tanh(roc_v)),float(np.tanh(cci_v))]
        scores_momentum=max(-1.0,min(1.0,sum(mom_vals)/5.0))
        # Volatility
        vol_val=(1.0-abs(bb-0.5)*2.0)*0.5+min(1.0,natr/0.5)*0.5
        scores_volatility=max(-1.0,min(1.0,vol_val))
        # Volume
        vr_score=1.0 if vr>1.2 else (-0.5 if vr<0.8 else 0.0)
        obv_n=ind.get("obv",0)/(abs(c)*1e8+1.0)
        vol_vals=[vr_score,float(np.tanh(obv_n)),mfi,cmf]
        scores_volume=max(-1.0,min(1.0,sum(vol_vals)/4.0))
        # Support/Resistance
        sr_scores=[]
        if s1>0: sr_scores.append(max(0.0,1.0-min(1.0,abs(c-s1)/s1/0.05)))
        if r1>0: sr_scores.append(max(0.0,1.0-min(1.0,abs(r1-c)/r1/0.05)))
        scores_sr=sum(sr_scores)/len(sr_scores) if sr_scores else 0.0
        # Structure
        scores_struct=min(1.0,(adx+(pdi+mdi)/200.0)/2.0)
        weights=[0.25,0.25,0.15,0.15,0.10,0.10]
        overall=scores_trend*weights[0]+scores_momentum*weights[1]+scores_volatility*weights[2]+scores_volume*weights[3]+scores_sr*weights[4]+scores_struct*weights[5]
        overall=max(-1.0,min(1.0,overall))
        comp={"trend":scores_trend,"momentum":scores_momentum,"volatility":scores_volatility,"volume":scores_volume,"support_resistance":scores_sr,"structure":scores_struct}
        return overall,comp


class MarketRegimeDetector:

    def detect_regime(self, df):
        c=np.asarray(df["close"],dtype=float)
        h=np.asarray(df["high"],dtype=float)
        lo=np.asarray(df["low"],dtype=float)
        v=np.asarray(df["volume"],dtype=float) if "volume" in df.columns else np.ones_like(c)
        ta=TechnicalAnalyzer()
        adx_a,pdi,mdi=ta.adx(h,lo,c)
        adx_v=float(adx_a[~np.isnan(adx_a)][-1]) if np.any(~np.isnan(adx_a)) else 0.0
        plus_v=float(pdi[~np.isnan(pdi)][-1]) if np.any(~np.isnan(pdi)) else 0.0
        minus_v=float(mdi[~np.isnan(mdi)][-1]) if np.any(~np.isnan(mdi)) else 0.0
        natr=ta.natr(h,lo,c)
        natr_v=float(natr[~np.isnan(natr)][-1]) if np.any(~np.isnan(natr)) else 0.0
        vr=float(v[-1]/(np.mean(v[-20:])+1e-10)) if len(v)>=20 else 1.0
        _,_,_,bb_p=ta.bollinger_bands(c)
        bb_pos=float(bb_p[~np.isnan(bb_p)][-1]) if np.any(~np.isnan(bb_p)) else 0.5
        rsi_v=ta.rsi(c)
        rsi_val=float(rsi_v[~np.isnan(rsi_v)][-1]) if np.any(~np.isnan(rsi_v)) else 50.0
        votes={}; confs=[]
        if adx_v>25:
            rg="BULL_TREND" if plus_v>minus_v else "BEAR_TREND"
            votes[rg]=votes.get(rg,0)+0.7; confs.append(min(1.0,adx_v/50))
        else:
            votes["SIDEWAYS"]=votes.get("SIDEWAYS",0)+0.6; confs.append(1.0-adx_v/25)
        if len(natr)>=20:
            vp=float(stats.percentileofscore(natr[~np.isnan(natr)],natr_v)/100)
            if vp>0.8: votes["HIGH_VOLATILITY"]=votes.get("HIGH_VOLATILITY",0)+0.6; confs.append(vp)
            elif vp<0.2: votes["LOW_VOLATILITY"]=votes.get("LOW_VOLATILITY",0)+0.5; confs.append(1.0-vp)
        if vr>1.5: votes["BREAKOUT"]=votes.get("BREAKOUT",0)+0.5; confs.append(min(1.0,vr/2.0))
        if bb_pos>0.95 or bb_pos<0.05: votes["REVERSAL"]=votes.get("REVERSAL",0)+0.5; confs.append(abs(bb_pos-0.5)*2)
        if rsi_val<25 or rsi_val>75: votes["REVERSAL"]=votes.get("REVERSAL",0)+0.7; confs.append(0.8)
        if not votes: return MarketRegime.UNKNOWN, 0.0
        dom=max(votes,key=votes.get)
        conf_val=np.mean(confs) if confs else 0.5
        return MarketRegime[dom], float(conf_val)
