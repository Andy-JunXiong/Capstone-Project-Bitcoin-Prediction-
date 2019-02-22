import * as d3 from 'd3'
import _ from 'lodash'

const colorScales = {
  m: d3.scaleOrdinal().range(['#4c5d91', '#4c92b9', '#53a488', '#a5ad5c']),
  f: d3.scaleOrdinal().range(['#a15599', '#d57599', '#b98c6f', '#e0da2f']),
  all: d3.scaleOrdinal().range(['#4c5d91', '#4c92b9', '#53a488', '#a5ad5c', '#a15599', '#d57599', '#b98c6f', '#e0da2f'])
}

const monthNames = [
  "Jan", "Feb", "Mar",
  "Apr", "May", "Jun", "Jul",
  "Aug", "Sep", "Oct",
  "Nov", "Dec"
]

export const years = _.range(0, 91)
export const timestart = new Date(2018, 6, 3)
console.log(timestart)

export function getDateString (days) {
  var date = new Date(timestart.valueOf());
  date.setDate(date.getDate() + days)
  var dateStr = monthNames[date.getMonth()] + ' ' + date.getDate() + ' ' + date.getFullYear()
  return dateStr
}

export function forenameColor (d) {
  return colorScales.all(d.name)
  // return colorScales[d.sex](d.forename)
}

export const fullRange = d3.extent(years)

export const defaultDuration = 750

export function maxPricesCount (forenames, range) {
  return _(forenames)
    .flatMap(d =>
      _(d.prices)
        .filter(({ year }) => year >= range.from && year <= range.to)
        .map('prices')
        .max()
    )
    .max() || 0
}

export const initialRange = { from: years[0], to: _.last(years) }
