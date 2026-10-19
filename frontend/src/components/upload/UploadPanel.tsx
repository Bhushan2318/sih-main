import { rich } from "../../i18n/rich";
import { translate } from "../../i18n";
import { useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { confirmMapping, uploadFile } from "../../api/upload";
import type { ConfirmMappingItem, UploadResponse } from "../../api/types";
import { ColumnMappingConfirmModal } from "./ColumnMappingConfirmModal";

type Phase = "idle" | "uploading" | "confirming" | "done" | "error";

export function UploadPanel() {
  const queryClient = useQueryClient();
  const inputRef = useRef<HTMLInputElement>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [pending, setPending] = useState<UploadResponse | null>(null);
  const [result, setResult] = useState<UploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ["modelStatus"] });
    queryClient.invalidateQueries({ queryKey: ["regions"] });
    queryClient.invalidateQueries({ queryKey: ["alerts"] });
    queryClient.invalidateQueries({ queryKey: ["ensemble"] });
    queryClient.invalidateQueries({ queryKey: ["replay"] });
  };

  const send = async (file: File) => {
    setPhase("uploading");
    setError(null);
    setResult(null);
    setPending(null);
    try {
      const res = await uploadFile(file);
      if (res.status === "pending_confirmation") {
        setPending(res);
        setPhase("confirming");
      } else {
        setResult(res);
        setPhase("done");
        invalidate();
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setPhase("error");
    }
  };

  const submitMappings = async (mappings: ConfirmMappingItem[]) => {
    if (!pending) return;
    setPhase("uploading");
    try {
      const res = await confirmMapping(pending.batch_id, mappings);
      setResult(res);
      setPending(null);
      setPhase("done");
      invalidate();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setPhase("error");
    }
  };

  return (
    <section className="card">
      <header className="card__head"><h3>{translate("up.title")}</h3></header>

      <div
        className={dragging ? "dropzone dropzone--over" : "dropzone"}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const f = e.dataTransfer.files?.[0];
          if (f) void send(f);
        }}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => { if (e.key === "Enter") inputRef.current?.click(); }}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".csv,.tsv,.txt,.xlsx,.xls,.json,.parquet"
          hidden
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void send(f);
            e.target.value = "";
          }}
        />
        {phase === "uploading" ? translate("up.working") : translate("up.drop")}
      </div>

      {phase === "error" && error ? <p className="notice notice--error">{error}</p> : null}

      {phase === "done" && result ? (
        <div className="upload-result">
          <p>
            {rich("up.ingested", { n: result.row_count_ingested.toLocaleString(), raw: result.row_count_raw.toLocaleString() }, [<b />])}
            {result.skipped_rows ? translate("up.skipped", { n: result.skipped_rows }) : ""}.
          </p>
          {result.canonical_variables_found.length ? (
            <p className="muted small">{translate("up.vars", { list: result.canonical_variables_found.join(", ") })}</p>
          ) : null}
          <p className="muted small">
            {translate("up.resolution", { pct: (result.region_resolution_rate * 100).toFixed(0) })}
            {result.source_profile_match !== "none" ? translate("up.reused", { match: result.source_profile_match }) : ""}
          </p>
          {result.status === "training_started" ? <p className="notice">{translate("up.training")}</p> : null}
          {result.notes.length ? (
            <details>
              <summary className="muted small">{translate("up.notes", { n: result.notes.length })}</summary>
              <ul className="notes">{result.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>
            </details>
          ) : null}
        </div>
      ) : null}

      {pending ? (
        <ColumnMappingConfirmModal
          upload={pending}
          onCancel={() => { setPending(null); setPhase("idle"); }}
          onSubmit={submitMappings}
        />
      ) : null}
    </section>
  );
}
