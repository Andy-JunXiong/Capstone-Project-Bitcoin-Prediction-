<style lang="stylus">
@import "./css/colors"

html
  height 100%
  margin 0
  padding 0

a:visited
  color: #6e4cb9

body
  font-family "HelveticaLTStd-Roman", sans-serif
  font-size 18px
  //width $width - 17px*2
  color color1
  background-color #fff

body, #app, content
  height 100%

content
  display: flex
  flex-direction: column
  padding-left: 19px

.title
  margin-bottom: 20px
  margin-top: 20px
  h1
    color: color-male
    font-family: "HelveticaLTStd-Bold"
    font-size: 35px
  h2
    font-size: 24px
    margin-bottom: -4px
  h1, h2
    margin: 0

.left-side
  // max-width 500px
  min-width 300px
  width 95%
  display: flex
  flex-direction: column
  margin-bottom: 10px
  margin-left: 30px

.right-side
  flex: 1

form.search
  width: 30%
  position: relative
  height: 32px
  > *
    position: absolute

  .search-image
    left: 4px
    top: 6px

  .clear
    top: 6px
    right: 0

  input
    color: #333
    font-family: "HelveticaLTStd-Roman"
    font-size: 19px
    padding-left 26px
    padding-top 4px
    width: calc(100% - 30px)
  a
    color: red
    font-family: sans-serif
    right: 0
    text-decoration: none

.forenames-list-container
  flex: 1
  overflow scroll
  margin-top 10px

ul.forenames-list
  list-style none
  padding 0
  margin 0
  display: flex
  flex-wrap: wrap
  // justify-content: space-between
  li
    cursor pointer
    padding 5px
    line-height 0.9
    display: inline-flex
    align-items: center
    &.m
      color color-male
    &.real
      color color-male  
    &.f
      color color-female
    &.pred
      color color-female
    &.selected
      line-height 1
      color white

.share-url
  cursor: crosshair

.bottom
  margin-bottom: 20px
  padding: 20px 95px 0 50px
  font-size: 12px
  color: color-male

  input[type=text]
    color: #333333
    width: 250px

  .block
    display flex
    justify-content: space-between
  a
    text-decoration: none
  .social-links > *
    margin-left 10px
  .social
    float: left
    margin-top: -9px
    width: 117px
    > *
      display: block
  .logo
    margin-top: -8px
    margin-left: -5px
    float: left
  .explanations
    margin-top: 1em
</style>

<template lang="pug">
#app
  content
    .left-side
      .title
        h2 From Jul 3rd 2018 to Sep 30th 2018 :
        h1 Bitcoin Price Prediction using Different Models
      form.search
        input(type="text" autocomplete="off" placeholder="Search" v-model="searchQuery")
        img.search-image(src="./images/search.png")
        a.clear(href="#" v-show="searchQuery.length" @click="searchQuery = ''")
          img(src="./images/clear.png")/
      .forenames-list-container
        ul.forenames-list(ref="forenamesList")
          li(v-for="forenameData in displayedForenames",
            :class="[forenameData.type, forenameData.selected ? 'selected' : '']",
            :style="forenameStyle(forenameData)",
            :title="forenameData.alternatives ? 'autres orthographes: ' + forenameData.alternatives.join(', ') : null",
            v-text="forenameData.name"
            @click.prevent="toggleForename(forenameData)"
          )
    .right-side
      graph(
        :forenames="selectedForenames",
        @forename:remove="toggleForename",
        @year-range="setYearRange"
      )
      //- .bottom
      //-   .block
      //-     .share-url
      //-       div Partagez ces résultats avec ce lien :
      //-       input(ref="share-url" type="text" @click="selectUrl", :value="shareUrl")
      //-     .social-links
      //-       iframe(src="https://www.facebook.com/plugins/share_button.php?href=http%3A%2F%2Fdataaddict.fr%2Fprenoms%2F&layout=button_count&size=large&mobile_iframe=true&appId=388906134841894&width=120&height=28" width="120" height="28" style="border:none;overflow:hidden" scrolling="no" frameborder="0" allowTransparency="true")
      //-       a.twitter-share-button(href="https://twitter.com/share" data-size="large" data-via="_Data_Addict_" data-lang="fr" data-hashtags="70ansdeprénoms")
      //-       a.github-button(href="https://github.com/dataaddict/prenoms" data-style="mega" aria-label="Star dataaddict/prenoms on GitHub") Star
      //-   .explanations
      //-     | Les prénoms sélectionnés sont les plus courants en France, ils ont été donnés au moins 2000 fois entre 1945 et 2015. Source :
      //-     a(href="https://www.insee.fr/fr/statistiques/2540004" target="_blank") Insee - Fichier des prénoms (Édition 2016)
</template>

<script>
import { forenameColor, initialRange, maxPricesCount, years } from './utils'
import * as d3 from 'd3'
import _ from 'lodash'
import graph from './graph.vue'

export default {
  name: 'app',
  components: { graph },
  data () {
    return {
      models: [],
      shareUrl: window.location.toString(),
      searchQuery: '',
      range: initialRange
    }
  },
  mounted () {
    console.log('mounted')
    const selectedIds = window.location.hash.slice(1).split(',')
    console.log(selectedIds)

    d3.json('models.json', (err, models) => {
      // window.data = data
      // window._ = _
      if (err) return window.alert(err)

      console.log('loaded', models.length)

      _.each(models, model => {
        model.selected = _.includes(selectedIds, model.id)
        model.style = {}
        model.prices = _.map(years, (year, i) => ({ year, prices: model.prices[i] }))
      })
      console.log(models)

      this.models = models
      this.$nextTick(window.onresize)
    })
  },

  watch: {
    forenames: 'refreshYearRange',
    selectedForenames (forenames) {
      window.location.hash = forenames.map(d => d.id).join(',')
      this.shareUrl = window.location.toString()
    }
  },

  methods: {
    toggleForename (forenameData) {
      forenameData.selected = !forenameData.selected
    },

    setYearRange (range) {
      this.range = range
      this.refreshYearRange()
    },

    refreshYearRange () {
      console.log('refreshYearRange')
      const overvallMax = maxPricesCount(this.forenames, this.range)

      const fontSizeScale = d3
        .scalePow()
        .exponent(0.8)
        .domain([0, overvallMax])
        .range([16, 40])

      _(this.forenames)
        .each(forenameData => {
          // const [fontSize, lineHeight] = sizes[level]
          const fontSize = fontSizeScale(maxPricesCount([forenameData], this.range)) + 'px'
          console.log(fontSize)
          forenameData.style = { fontSize }
        })
    },

    forenameStyle (forenameData) {
      const backgroundColor = forenameData.selected ? forenameColor(forenameData) : ''
      return _.assign({ backgroundColor }, forenameData.style)
    },

    // selectUrl () {
    //   // http://stackoverflow.com/a/1173319
    //   const el = this.$refs['share-url']
    //
    //   if (document.selection) {
    //     var range = document.body.createTextRange()
    //     range.moveToElementText(el)
    //     range.select()
    //   } else if (window.getSelection) {
    //     const range = document.createRange()
    //     range.selectNode(el)
    //     window.getSelection().addRange(range)
    //   }
    // }
    selectUrl () {
      this.$refs['share-url'].select()
    }
  },

  computed: {
    selectedForenames () {
      return _.filter(this.models, 'selected')
    },
    displayedForenames () {
      if (this.searchQuery === '') return this.models
      const sanitize = s => _.deburr(s).toLowerCase()
      const q = sanitize(this.searchQuery)

      return this.models.filter(d => sanitize(d.name).includes(q))
    }
  }
}
</script>
