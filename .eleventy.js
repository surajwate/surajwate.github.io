const pluginDate = require("eleventy-plugin-date");
const markdownIt = require("markdown-it");
const syntaxHighlight = require("@11ty/eleventy-plugin-syntaxhighlight");
const { JSDOM } = require("jsdom");
const CleanCSS = require("clean-css");
const Terser = require("terser");


module.exports = function (eleventyConfig) {
  // Copy CNAME file to output directory
  eleventyConfig.addPassthroughCopy("CNAME");
  // Add syntax highlighting
  eleventyConfig.addPlugin(syntaxHighlight);

  // Pass through sitemap to output directory
  eleventyConfig.addPassthroughCopy("sitemap.xml");

  // Markdown options

  let options = {
    html: true,
    breaks: true,
    linkify: true,
    typographer: true,
  };
  eleventyConfig.setLibrary("md", markdownIt(options));

  // Copy `assets/` to `_site/assets`
  eleventyConfig.addPassthroughCopy("src/assets");

  // Blog collection sorted by date
  eleventyConfig.addCollection("posts", function (collection) {
    return collection.getFilteredByGlob("src/blog/posts/**/*.md").filter(post => !post.data.draft).sort((a, b) => {
      return b.date - a.date; // Sort by date in descending order
    });
  });

  // Projects collection sorted by date
  eleventyConfig.addCollection("projects", function (collection) {
    return collection.getFilteredByGlob("src/projects/**/*.md").filter(project => !project.data.draft).sort((a, b) => {
      return b.date - a.date; // Sort by date in descending order
    });
  });

  // --- Date Plugin ---
  eleventyConfig.addPlugin(pluginDate, {
    // Specify custom date formats
    formats: {
      // Change the readableDate filter to use abbreviated months.
      readableDate: { year: "numeric", month: "short", day: "numeric" },
      // Add a new filter to format a Date to just the month and year.
      readableMonth: { year: "numeric", month: "long" },
      // Add a new filter using formatting tokens.
      timeZone: "z",
    }
  });

  // --- Lazy Loading Transform ---
  eleventyConfig.addTransform("lazyload", function (content, outputPath) {
    if (outputPath && outputPath.endsWith(".html")) {
      let dom = new JSDOM(content);
      let images = dom.window.document.querySelectorAll("img");

      // Add loading="lazy" to all <img> tags
      images.forEach(img => {
        img.setAttribute("loading", "lazy");
      });

      return dom.serialize();
    }

    return content;
  });

  // --- Minify CSS ---
  eleventyConfig.addTransform("cssmin", function (content, outputPath) {
    if (outputPath.endsWith(".css")) {
      let minified = new CleanCSS({}).minify(content).styles;
      return minified;
    }
    return content;
  });

  // --- Minify JS ---
  eleventyConfig.addTransform("jsmin", async function (content, outputPath) {
    if (outputPath.endsWith(".js")) {
      let minified = await Terser.minify(content);
      return minified.code;
    }
    return content;
  });

  return {
    dir: {
      input: "src",
      output: "_site",
      includes: "_includes",
    },
    templateFormats: ["html", "md", "njk"],
    htmlTemplateEngine: "njk",
    markdownTemplateEngine: "njk",
    dataTemplateEngine: "njk",
  };
}