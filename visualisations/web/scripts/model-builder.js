const csv = require('csvtojson')
const fs = require('fs')
const rawPriceDataBasePath = './raw-data/prices/'
const rawAcfDataBasePath = './raw-data/autocorrelations/'
const rmseDataFilePath = './raw-data/rmse/metrics.csv'

const modelNames = [
    'ARIMA',
    'GRU',
    'LSTM',
    'MLP',
    'ExtraTrees',
    'XGBoost',
    'DecisionTree',
    'SupportVectorRegression',
    'KNN',
    'BayesianRidge',
    'ElasticNet',
    'Ridge',
    'Lasso',
    'LinearRegression'
]
const realPriceName = ['Y_test']

const processModels = async (modelNames) => {
    const errArray = await csv().fromFile(rmseDataFilePath)
    const rmseData = errArray.find(data => data.Type = 'RMSE')
    const modelsData = []
    for (const modelName of modelNames) {
        const rawPriceData = await csv().fromFile(rawPriceDataBasePath + modelName + '.csv')
        const rawAcfData = await csv().fromFile(rawAcfDataBasePath + modelName + '.csv')
        modelsData.push({
            id: modelName.toLowerCase(),
            name: modelName,
            type: 'pred',
            rmse: rmseData[modelName],
            prices: rawPriceData.map(price => parseFloat(price.Y_pred)),
            acfs: rawAcfData.map(acf => parseFloat(acf.Y))
        })
    }
    rawRealPriceData = await csv().fromFile(rawPriceDataBasePath + realPriceName[0] + '.csv')
    modelsData.push({
        id: realPriceName[0].toLowerCase(),
        name: realPriceName[0],
        type: 'real',
        rmse: 0,
        prices: rawRealPriceData.map(price => parseFloat(price.Y_test)),
        acfs: []
    })
    return modelsData
}

const main = async () => {
    const modelsData = await processModels(modelNames)
    fs.writeFileSync('./src/data/models.json', JSON.stringify(modelsData, null, 4));  
}

main()
