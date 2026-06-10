# 公開（デプロイ）ガイド

このアプリは **サーバー不要・全計算がブラウザ内で完結する静的サイト** です。
`npm run build` で生成される `dist/` フォルダを置くだけで、誰でもURLからアクセスできます。
（画面遷移はアプリ内部の状態で管理しており、URLルーティングが無いので 404 設定なども不要です）

---

## いちばん簡単：Netlify Drop（アカウント不要・約1分）

1. ビルドする
   ```
   npm install
   npm run build
   ```
2. ブラウザで <https://app.netlify.com/drop> を開く
3. 生成された **`dist` フォルダごと** ページにドラッグ＆ドロップ
4. すぐに `https://〇〇〇.netlify.app` のような **公開URL** が発行されます

> 無料アカウントを作ると、そのURLを永続化したり独自ドメインを設定できます。
> 更新したいときは、再度 `npm run build` して `dist` をドラッグし直すだけです。

---

## 永続URL＆自動更新：GitHub Pages（無料）

GitHubリポジトリに push するたびに自動でビルド＆公開されます
（`.github/workflows/deploy.yml` を同梱済み）。

1. このフォルダを GitHub リポジトリにする
   ```
   git init
   git add .
   git commit -m "init"
   git branch -M main
   git remote add origin https://github.com/<ユーザー名>/<リポジトリ名>.git
   git push -u origin main
   ```
2. GitHub のリポジトリ画面 → **Settings → Pages** を開く
3. **Build and deployment → Source** を **「GitHub Actions」** に設定
4. 数分後、`https://<ユーザー名>.github.io/<リポジトリ名>/` で公開されます

> 以降は `git push` するだけで自動的にサイトが更新されます。
> `vite.config.ts` の `base: './'`（相対パス）を設定済みなので、
> リポジトリ名に関係なくそのまま動きます。

---

## Vercel

1. <https://vercel.com> でリポジトリを連携（Import）
2. フレームワークは自動検出（Vite）。そのまま Deploy
3. `https://〇〇〇.vercel.app` が発行されます

---

## 自前のサーバー / S3 など

`npm run build` 後の **`dist/` の中身をそのままアップロード** するだけです。
特別なサーバー設定は不要です（`python -m http.server` などでも配信できます）。

---

## ローカル確認

公開前に本番ビルドをローカルで確認できます。
```
npm run build
npm run preview
```
表示された `http://localhost:4173` を開いて動作を確認してください。
