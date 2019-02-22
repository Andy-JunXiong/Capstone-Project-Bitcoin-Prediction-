const Binance = require('binance-api-node').default
const moment = require('moment')
const fs = require('fs')
const json2csv = require('json2csv').parse;

require('dotenv').config()

const client = Binance()

// Authenticated client, can make signed calls
const client2 = Binance({
  apiKey: process.env.API_KEY,
  apiSecret: process.env.API_SECRET,
})

const SYMBOLS = [
  'BTCUSDT',
  'ETHUSDT',
  'XRPUSDT',
  'LTCUSDT'
]

const INTERVALS = [
  // '1h',
  '1d'
]

const startTime = moment('01-01-2016', 'MM-DD-YYYY').unix() * 1000
const endTime = moment('01-01-2019', 'MM-DD-YYYY').unix() * 1000

const fetchCandleSticks = async (symbol, interval, startTime, endTime) => {
  return await client.candles({
    symbol,
    interval,
    startTime,
    endTime
  })
}

const fields = ['openTime', 'open', 'high', 'low', 'close', 'volume', 'closeTime', 'quoteVolume', 'trades', 'baseAssetVolume', 'quoteAssetVolume'];
const opts = { fields };

const writeDataToCsv = async () => {
  SYMBOLS.forEach(async (symbol) => {
    INTERVALS.forEach(async (interval) => {
      let data = await fetchCandleSticks(symbol, interval, startTime, endTime)
      try {
        const csv = json2csv(data, opts);
        fs.writeFileSync(`./dist/candlestick_${symbol}_${interval}_${startTime}_${endTime}.csv`, csv, {
          encoding: 'utf-8'
        })
      } catch (err) {
        console.error(err);
      }
    })
  })
}

writeDataToCsv()
